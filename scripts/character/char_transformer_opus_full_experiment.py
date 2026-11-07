import argparse, gc, json, math, os, random, time
from collections import Counter
from pathlib import Path
import evaluate, numpy as np, pandas as pd, torch, torch.nn as nn
from comet import download_model, load_from_checkpoint
from scipy import stats
from sentence_transformers import SentenceTransformer, util
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def read_lines(path):
    p = Path(path)
    if not p.is_file(): raise FileNotFoundError(path)
    return [x.rstrip("\r\n") for x in p.open("r", encoding="utf-8-sig")]


def load_parallel(src, tgt, limit=None):
    s, t = read_lines(src), read_lines(tgt)
    if len(s) != len(t):
        raise ValueError(f"Line mismatch: {src}={len(s)}, {tgt}={len(t)}")
    df = pd.DataFrame({"source": s, "target": t}).dropna()
    df = df[(df.source.str.strip() != "") & (df.target.str.strip() != "")]
    df = df.drop_duplicates().reset_index(drop=True)
    return df.head(limit).reset_index(drop=True) if limit else df


class CharTokenizer:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq; self.stoi = {}; self.itos = {}

    def fit(self, texts):
        c = Counter(ch for text in texts for ch in text)
        vocab = SPECIAL + sorted(ch for ch, n in c.items() if n >= self.min_freq)
        self.stoi = {tok: i for i, tok in enumerate(vocab)}
        self.itos = {i: tok for tok, i in self.stoi.items()}

    @property
    def pad_id(self): return self.stoi["<pad>"]
    @property
    def bos_id(self): return self.stoi["<bos>"]
    @property
    def eos_id(self): return self.stoi["<eos>"]
    @property
    def unk_id(self): return self.stoi["<unk>"]
    @property
    def vocab_size(self): return len(self.stoi)

    def encode(self, text, max_len):
        ids = [self.bos_id] + [self.stoi.get(ch, self.unk_id) for ch in text] + [self.eos_id]
        ids = ids[:max_len]
        ids[-1] = self.eos_id
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            tok = self.itos.get(int(i), "<unk>")
            if tok == "<eos>": break
            if tok not in SPECIAL: out.append(tok)
        return "".join(out)

    def save(self, folder):
        Path(folder).mkdir(parents=True, exist_ok=True)
        Path(folder, "tokenizer.json").write_text(json.dumps({"min_freq": self.min_freq, "stoi": self.stoi}, ensure_ascii=False, indent=2), encoding="utf-8")


class ParallelDataset(Dataset):
    def __init__(self, df, src_tok, tgt_tok, max_src, max_tgt):
        self.df, self.src_tok, self.tgt_tok = df, src_tok, tgt_tok
        self.max_src, self.max_tgt = max_src, max_tgt
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return {
            "src": torch.tensor(self.src_tok.encode(str(r.source), self.max_src)),
            "tgt": torch.tensor(self.tgt_tok.encode(str(r.target), self.max_tgt)),
            "src_text": str(r.source), "tgt_text": str(r.target)
        }


def make_collate(src_pad, tgt_pad):
    def collate(batch):
        return {
            "src": pad_sequence([x["src"] for x in batch], batch_first=True, padding_value=src_pad),
            "tgt": pad_sequence([x["tgt"] for x in batch], batch_first=True, padding_value=tgt_pad),
            "src_text": [x["src_text"] for x in batch],
            "tgt_text": [x["tgt_text"] for x in batch],
        }
    return collate


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=4096):
        super().__init__(); self.drop = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return self.drop(x + self.pe[:, :x.size(1)])


class CharTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, src_pad, tgt_pad, d_model=512, heads=8, enc_layers=6, dec_layers=6, ff=2048, dropout=0.1, max_len=4096):
        super().__init__(); self.d_model=d_model; self.src_pad=src_pad; self.tgt_pad=tgt_pad
        self.src_emb=nn.Embedding(src_vocab,d_model,padding_idx=src_pad)
        self.tgt_emb=nn.Embedding(tgt_vocab,d_model,padding_idx=tgt_pad)
        self.src_pos=PositionalEncoding(d_model,dropout,max_len); self.tgt_pos=PositionalEncoding(d_model,dropout,max_len)
        self.tf=nn.Transformer(d_model=d_model,nhead=heads,num_encoder_layers=enc_layers,num_decoder_layers=dec_layers,dim_feedforward=ff,dropout=dropout,batch_first=True,norm_first=True)
        self.out=nn.Linear(d_model,tgt_vocab)
        for p in self.parameters():
            if p.dim()>1: nn.init.xavier_uniform_(p)

    def encode(self, src):
        src_mask = src.eq(self.src_pad)
        mem = self.tf.encoder(self.src_pos(self.src_emb(src)*math.sqrt(self.d_model)), src_key_padding_mask=src_mask)
        return mem, src_mask

    def decode(self, tgt, mem, src_mask):
        causal = torch.triu(torch.ones(tgt.size(1),tgt.size(1),device=tgt.device,dtype=torch.bool),diagonal=1)
        x = self.tf.decoder(self.tgt_pos(self.tgt_emb(tgt)*math.sqrt(self.d_model)), mem, tgt_mask=causal, tgt_key_padding_mask=tgt.eq(self.tgt_pad), memory_key_padding_mask=src_mask)
        return self.out(x)

    def forward(self, src, tgt):
        mem, mask = self.encode(src); return self.decode(tgt, mem, mask)


def save_ckpt(path, model, opt, sched, scaler, epoch, step, best, config, history):
    torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"sched":sched.state_dict(),"scaler":scaler.state_dict(),"epoch":epoch,"step":step,"best":best,"config":config,"history":history}, path)


def latest_ckpt(folder):
    c=[]
    for p in Path(folder).glob("checkpoint_*.pt"):
        try: c.append((int(p.stem.split("_")[-1]), p))
        except: pass
    return str(sorted(c)[-1][1]) if c else None


def eval_loss(model, loader, criterion, device, amp):
    model.eval(); total=0; tokens=0
    with torch.inference_mode():
        for b in tqdm(loader, desc="Validation", leave=False):
            s,t=b["src"].to(device),b["tgt"].to(device); inp,lab=t[:,:-1],t[:,1:]
            with torch.autocast("cuda",dtype=torch.float16,enabled=amp):
                logits=model(s,inp); loss=criterion(logits.reshape(-1,logits.size(-1)),lab.reshape(-1))
            n=lab.ne(model.tgt_pad).sum().item(); total+=loss.item()*n; tokens+=n
    return total/max(tokens,1)


def train(model, train_loader, dev_loader, args, config, device):
    ckdir=Path(args.output_dir,"checkpoints"); ckdir.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,betas=(0.9,0.98),eps=1e-9,weight_decay=args.weight_decay)
    updates=math.ceil(len(train_loader)/args.gradient_accumulation_steps); total_steps=updates*args.epochs
    def lrfn(step):
        if step<args.warmup_steps:return max(step,1)/max(args.warmup_steps,1)
        return max(total_steps-step,0)/max(total_steps-args.warmup_steps,1)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lrfn)
    scaler=torch.cuda.amp.GradScaler(enabled=args.fp16 and torch.cuda.is_available())
    criterion=nn.CrossEntropyLoss(ignore_index=model.tgt_pad,label_smoothing=args.label_smoothing)
    start_epoch=0; step=0; best=float("inf"); history=[]; stale=0
    if args.resume_from_checkpoint:
        p=latest_ckpt(ckdir)
        if p:
            z=torch.load(p,map_location=device); model.load_state_dict(z["model"]); opt.load_state_dict(z["opt"]); sched.load_state_dict(z["sched"]); scaler.load_state_dict(z["scaler"])
            start_epoch=z["epoch"]; step=z["step"]; best=z["best"]; history=z.get("history",[]); print("Resuming:",p)
    best_path=Path(args.output_dir,"best_model.pt")
    for epoch in range(start_epoch,args.epochs):
        model.train(); opt.zero_grad(set_to_none=True); total=0; tokens=0; started=time.time()
        bar=tqdm(enumerate(train_loader),total=len(train_loader),desc=f"Epoch {epoch+1}/{args.epochs}")
        for i,b in bar:
            s,t=b["src"].to(device,non_blocking=True),b["tgt"].to(device,non_blocking=True); inp,lab=t[:,:-1],t[:,1:]
            with torch.autocast("cuda",dtype=torch.float16,enabled=args.fp16 and torch.cuda.is_available()):
                logits=model(s,inp); loss=criterion(logits.reshape(-1,logits.size(-1)),lab.reshape(-1))/args.gradient_accumulation_steps
            scaler.scale(loss).backward(); n=lab.ne(model.tgt_pad).sum().item(); total+=loss.item()*args.gradient_accumulation_steps*n; tokens+=n
            if (i+1)%args.gradient_accumulation_steps==0 or i+1==len(train_loader):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step+=1
                bar.set_postfix(loss=total/max(tokens,1),lr=sched.get_last_lr()[0],step=step)
                if args.save_steps and step%args.save_steps==0:
                    save_ckpt(Path(ckdir,f"checkpoint_{step}.pt"),model,opt,sched,scaler,epoch,step,best,config,history)
                    old=sorted(ckdir.glob("checkpoint_*.pt"),key=lambda x:int(x.stem.split("_")[-1]))[:-args.keep_checkpoints]
                    for q in old:q.unlink(missing_ok=True)
        tr=total/max(tokens,1); va=eval_loss(model,dev_loader,criterion,device,args.fp16 and torch.cuda.is_available())
        row={"epoch":epoch+1,"global_step":step,"train_loss":tr,"validation_loss":va,"learning_rate":sched.get_last_lr()[0],"epoch_seconds":time.time()-started}; history.append(row)
        pd.DataFrame(history).to_excel(Path(args.output_dir,"training_history.xlsx"),index=False)
        if va<best:
            best=va; stale=0; torch.save({"model":model.state_dict(),"config":config,"best_validation_loss":best,"epoch":epoch+1,"step":step},best_path); print("Saved best model")
        else:
            stale+=1; print(f"No improvement ({stale}/{args.early_stopping_patience})")
        save_ckpt(Path(ckdir,f"checkpoint_{step}.pt"),model,opt,sched,scaler,epoch+1,step,best,config,history)
        if stale>=args.early_stopping_patience: print("Early stopping"); break
    return best_path


def greedy(model, src, tok, max_len):
    mem,mask=model.encode(src); out=torch.full((src.size(0),1),tok.bos_id,dtype=torch.long,device=src.device); done=torch.zeros(src.size(0),dtype=torch.bool,device=src.device)
    for _ in range(max_len-1):
        nxt=model.decode(out,mem,mask)[:,-1].argmax(-1,keepdim=True); out=torch.cat([out,nxt],1); done|=nxt.squeeze(1).eq(tok.eos_id)
        if bool(done.all()): break
    return out


def generate(model, loader, tok, max_len, device):
    model.eval(); ss=[]; rr=[]; pp=[]
    with torch.inference_mode():
        for b in tqdm(loader,desc="Generating translations"):
            ids=greedy(model,b["src"].to(device),tok,max_len)
            pp.extend(tok.decode(x.tolist()) for x in ids); ss.extend(b["src_text"]); rr.extend(b["tgt_text"])
    return ss,rr,pp


def stats_row(name, vals):
    vals=np.asarray(vals,dtype=float); vals=vals[np.isfinite(vals)]; mean=float(vals.mean()); std=float(vals.std(ddof=1)) if len(vals)>1 else 0.; sem=float(stats.sem(vals)) if len(vals)>1 else 0.; lo,hi=(stats.t.interval(.95,len(vals)-1,loc=mean,scale=sem) if len(vals)>1 else (mean,mean))
    return {"Metric":name,"N":len(vals),"Mean":mean,"Std Dev":std,"Standard Error":sem,"95% CI Lower":float(lo),"95% CI Upper":float(hi)}


def metrics(sources,preds,refs,device,args):
    bleu=evaluate.load("sacrebleu"); chrf=evaluate.load("chrf")
    corpus_bleu=bleu.compute(predictions=preds,references=[[r] for r in refs])["score"]
    corpus_chrf=chrf.compute(predictions=preds,references=refs)["score"]
    sb=[]; sc=[]
    for p,r in tqdm(zip(preds,refs),total=len(preds),desc="Sentence BLEU/chrF"):
        sb.append(bleu.compute(predictions=[p],references=[[r]])["score"]); sc.append(chrf.compute(predictions=[p],references=[r])["score"])
    sbert=SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2",device=device)
    re=sbert.encode(refs,convert_to_tensor=True,show_progress_bar=True,batch_size=args.sbert_batch_size); pe=sbert.encode(preds,convert_to_tensor=True,show_progress_bar=True,batch_size=args.sbert_batch_size)
    ss=util.cos_sim(pe,re).diagonal().detach().cpu().numpy(); del sbert,re,pe; clear_memory()
    cm=load_from_checkpoint(download_model(args.comet_model_name)); out=cm.predict([{"src":s,"mt":p,"ref":r} for s,p,r in zip(sources,preds,refs)],batch_size=args.comet_batch_size,gpus=1 if device=="cuda" else 0,progress_bar=True); cs=np.asarray(out.scores); del cm; clear_memory()
    per=pd.DataFrame({"Sentence_BLEU":sb,"Sentence_chrF":sc,"SBERT":ss,"COMET":cs})
    summary={"BLEU":float(corpus_bleu),"chrF":float(corpus_chrf),"SBERT":float(ss.mean()),"COMET":float(cs.mean())}
    stat=pd.DataFrame([stats_row(c,per[c].values) for c in per.columns])
    return summary,per,stat


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--direction",required=True,choices=["ar-en","en-ar"])
    for n in ["train_ar","train_en","dev_ar","dev_en","test_ar","test_en"]: p.add_argument(f"--{n}",required=True)
    p.add_argument("--output_dir",default="char_transformer_outputs"); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch_size",type=int,default=16); p.add_argument("--eval_batch_size",type=int,default=16); p.add_argument("--gradient_accumulation_steps",type=int,default=2)
    p.add_argument("--learning_rate",type=float,default=5e-4); p.add_argument("--weight_decay",type=float,default=.01); p.add_argument("--label_smoothing",type=float,default=.1); p.add_argument("--max_grad_norm",type=float,default=1.); p.add_argument("--warmup_steps",type=int,default=4000)
    p.add_argument("--max_source_len",type=int,default=512); p.add_argument("--max_target_len",type=int,default=512); p.add_argument("--max_gen_len",type=int,default=512)
    p.add_argument("--embedding_dim",type=int,default=512); p.add_argument("--num_heads",type=int,default=8); p.add_argument("--encoder_layers",type=int,default=6); p.add_argument("--decoder_layers",type=int,default=6); p.add_argument("--feedforward_dim",type=int,default=2048); p.add_argument("--dropout",type=float,default=.1); p.add_argument("--min_char_frequency",type=int,default=1)
    p.add_argument("--save_steps",type=int,default=1000); p.add_argument("--keep_checkpoints",type=int,default=3); p.add_argument("--early_stopping_patience",type=int,default=3); p.add_argument("--num_workers",type=int,default=0)
    p.add_argument("--max_train_samples",type=int); p.add_argument("--max_dev_samples",type=int); p.add_argument("--max_test_samples",type=int)
    p.add_argument("--sbert_batch_size",type=int,default=16); p.add_argument("--comet_batch_size",type=int,default=8); p.add_argument("--comet_model_name",default="Unbabel/wmt22-comet-da"); p.add_argument("--fp16",action="store_true"); p.add_argument("--resume_from_checkpoint",action="store_true"); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if a.embedding_dim%a.num_heads: p.error("embedding_dim must be divisible by num_heads")
    return a


def main():
    a=parse_args(); seed_all(a.seed); os.makedirs(a.output_dir,exist_ok=True); device="cuda" if torch.cuda.is_available() else "cpu"; print("Device:",device)
    if torch.cuda.is_available(): print("GPU:",torch.cuda.get_device_name(0),f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    if a.direction=="ar-en":
        tr=load_parallel(a.train_ar,a.train_en,a.max_train_samples); dv=load_parallel(a.dev_ar,a.dev_en,a.max_dev_samples); te=load_parallel(a.test_ar,a.test_en,a.max_test_samples)
    else:
        tr=load_parallel(a.train_en,a.train_ar,a.max_train_samples); dv=load_parallel(a.dev_en,a.dev_ar,a.max_dev_samples); te=load_parallel(a.test_en,a.test_ar,a.max_test_samples)
    st,tt=CharTokenizer(a.min_char_frequency),CharTokenizer(a.min_char_frequency); st.fit(tr.source.tolist()); tt.fit(tr.target.tolist()); st.save(Path(a.output_dir,"source_tokenizer")); tt.save(Path(a.output_dir,"target_tokenizer"))
    print("Vocab sizes:",st.vocab_size,tt.vocab_size)
    collate=make_collate(st.pad_id,tt.pad_id)
    train_loader=DataLoader(ParallelDataset(tr,st,tt,a.max_source_len,a.max_target_len),batch_size=a.batch_size,shuffle=True,collate_fn=collate,num_workers=a.num_workers,pin_memory=torch.cuda.is_available())
    dev_loader=DataLoader(ParallelDataset(dv,st,tt,a.max_source_len,a.max_target_len),batch_size=a.eval_batch_size,shuffle=False,collate_fn=collate,num_workers=a.num_workers,pin_memory=torch.cuda.is_available())
    test_loader=DataLoader(ParallelDataset(te,st,tt,a.max_source_len,a.max_target_len),batch_size=a.eval_batch_size,shuffle=False,collate_fn=collate,num_workers=a.num_workers,pin_memory=torch.cuda.is_available())
    config={"src_vocab":st.vocab_size,"tgt_vocab":tt.vocab_size,"src_pad":st.pad_id,"tgt_pad":tt.pad_id,"d_model":a.embedding_dim,"heads":a.num_heads,"enc_layers":a.encoder_layers,"dec_layers":a.decoder_layers,"ff":a.feedforward_dim,"dropout":a.dropout,"max_len":max(a.max_source_len,a.max_target_len,a.max_gen_len)+16}
    Path(a.output_dir,"model_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    model=CharTransformer(**config).to(device); print("Parameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))
    best=train(model,train_loader,dev_loader,a,config,device); z=torch.load(best,map_location=device); model.load_state_dict(z["model"]); model.to(device)
    sources,refs,preds=generate(model,test_loader,tt,a.max_gen_len,device); model.to("cpu"); clear_memory(); summary,per,stat=metrics(sources,preds,refs,device,a)
    out=pd.DataFrame({"direction":a.direction,"source":sources,"reference":refs,"prediction":preds}); out=pd.concat([out,per],axis=1)
    trans=Path(a.output_dir,f"character_transformer_{a.direction}_translations_with_metrics.xlsx"); stfile=Path(a.output_dir,f"character_transformer_{a.direction}_statistics.xlsx"); out.to_excel(trans,index=False); stat.to_excel(stfile,index=False)
    final=pd.DataFrame([{**{"experiment":"character_transformer_from_scratch","direction":a.direction,"source_vocab_size":st.vocab_size,"target_vocab_size":tt.vocab_size,"best_validation_loss":z["best_validation_loss"],"translations_excel":str(trans),"statistics_excel":str(stfile),"best_model":str(best)},**summary}]); final.to_excel(Path(a.output_dir,"final_results.xlsx"),index=False)
    print("Done",summary)


if __name__=="__main__": main()
