def get_stats(ids):
    counts={}
    for pair in zip(ids,ids[1:]):
        counts[pair]=counts.get(pair,0)+1#counts the no of times a pair appears in the vocab
    return counts

def merge(ids,pair, new_id):
    out,i=[],0
    while i<len(ids):
        if i<len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]:#if the checks if the consecutive tokens in the ids is a part of the pair
            out.append(new_id)
            i+=2
        else:
            out.append(ids[i])
            i+=1
    return out

#training the bpe tokeniser
# this is the code that is used to train the bpe tokeniser

def train_bpe(text,vocab_size):
    ids=list(text.encode('utf-8'))
    merges={}
    vocab={i:bytes([i]) for i in range(256)}
    for i in range(vocab_size-256):
        stats=get_stats(ids)
        if not stats:
            break
        pair=max(stats,key=lambda p: (stats[p],-p[0],-p[1]))#gets the pair that appears the maximum times amongst the pairs
        new_id=256+i
        ids=merge(ids,pair,new_id)
        merges[pair]=new_id
        vocab[new_id]=vocab[pair[0]]+vocab[pair[1]]
    return merges,vocab


def encode(text, merges):
    ids = list(text.encode("utf-8"))#encodes the text into the utf-8 foramt

    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))#finds out and merges the lowest ranked token pair

        if pair not in merges:# if there is no valid pair left to merge then it stops the loop and no furhter merging occurs

            break
        ids = merge(ids, pair, merges[pair])
    return ids


def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
    #the code above is responsible for decoding the merged part back to the vocabulary
    #basically a one line decode function for the utf-8 format 


#end to end bpe tokenisation example 
# 1. Corpus definition
corpus = "low lower newest widest" * 10

# 2. Train tokenizer to learn up to 260 vocab items
merges, vocab = train_bpe(corpus, vocab_size=260)

# 3. Encode new text
input_text = "lowest"
encoded_ids = encode(input_text, merges)

# 4. Decode back to text
decoded_text = decode(encoded_ids, vocab)

print(f"Encoded IDs: {encoded_ids}")
print(f"Decoded Text: {decoded_text}")
assert decoded_text == input_text


if __name__ == "__main__":
    # end to end bpe tokenisation example
    corpus = "low lower newest widest" * 10
    merges, vocab = train_bpe(corpus, vocab_size=260)
    input_text = "lowest"
    encoded_ids = encode(input_text, merges)
    decoded_text = decode(encoded_ids, vocab)

    print(f"Encoded IDs: {encoded_ids}")
    print(f"Decoded Text: {decoded_text}")
    assert decoded_text == input_text
    print("BPE demo successful.")
