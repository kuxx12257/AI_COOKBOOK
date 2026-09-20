import numpy as np
def pack_documents(docs,encode_fn, eos_id):
    #docs is the list os raw socument strings
    #encode_fn is the encoding function
    #eos id is the vocabualry's end of sequence token id
    assert eos_id<65536,"vocab exceeds unit16, ids would silently wrap"

    out=[]

    for d in docs:
        out.extend(encode_fn(d))
        out.append(eos_id)
        #scans each of the documents or the piece of each voczb and immediately appends its eos token into the list out
    return np.array(out, dtype=np.uint16)

def get_batch(data, batch_size, context_len, rng):
    #data is the 1d array of all the strings of text
    #batch_size as we know is fixe
    #context len is also fixed 
    #reg is the randomly generated number using numpy
    ix=rng.integers(0,len(data)-context_len-1,size=batch_size)
    x=np.stack([data[i  :i+context_len].astype(np.int64) for i in ix])
    y=np.stack([data[i+1:i+context_len+1].astype(np.int64)for i in ix])
    #forms the tensor of 64 the x and y inputs for the self supervised learning


    return x,y

assert(x[:, 1:]==y[:, :1]).all(),"INPUT/TARGET SHIFT IS WRONG "
#checks if the x values and y values are upto the mark and the offset must be upto the mark
#this prevents catastrophic unshifted target leaks

def effective_context(data,eos_id,context_len,n=20000,seed=0):
    #this function is used for samplng n random points in data to measure the axtual amount of the same -document history available in a context _len window
    for p in pos:
        window=data[p-context_len:p]
        hits=np.nonzero(window==eos_id)[0]
        dists.append(context_len-1-hits[-1] if len(hits) else context_len)
    return np.array(dists)

rng=np.random.default_rng(0)
docs=[rng.integers(1,8000,size=int(rng.normal(250,60))).tolist() for _ in range(4000)]
#used for rando, integer generation and generates 4000 dummy documents forma a gamma distribution
#mimics tiny stories
data=pack_documents(docs, lambda d: d,eos_id=8192)
print(f"packed:{len(data):,} tokens, {data.nbytes/1e6:.1f} MB as uint16")
print(f"    would be{len(data)*8/1e6:.1f} MB as int64")
#packs the simulated documents with an identity encoder lambda d:d and eos_id=8191 
#pritns the packed array length and compares the byte size of uint16 vs int 64

x, y = get_batch(data, batch_size=4, context_len=512, rng=np.random.default_rng(1))
assert_shift_is_correct(x, y)
print(f"x {x.shape} {x.dtype}   shift invariant: OK")

#this is responsibl efor generating a batch size of teh required value

d = effective_context(data, eos_id=8191, context_len=512)
print(f"\neffective context: mean {d.mean():.0f}  median {np.median(d):.0f}  "
      f"p90 {np.percentile(d,90):.0f}  frac full-512 {(d==512).mean():.3f}")

#runs the efectiv context over the data and prints the important statistics

def get_batch_memmao(path, batch_size, context_len, rng):
    data=np.memmap(path,dtype=np.unit16, mode="r")
    ix=rng.intehers(0,len(data)-context_len-1, size=batch_size)
    x = np.stack([data[i     : i + context_len    ].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1 : i + context_len + 1].astype(np.int64) for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)




