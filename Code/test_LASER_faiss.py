from laser_encoders import LaserEncoderPipeline
import faiss

lv_texts = ["Sveiki draugs"]
en_texts = ["Hi friend", "I like apples", "My name is John", "The weather is very good today!", "Phone is ringing"]

encoder_en = LaserEncoderPipeline(lang="eng_Latn")
encoder_lv = LaserEncoderPipeline(lang="lvs_Latn")
embeddings_lv = encoder_lv.encode_sentences(lv_texts)
embeddings_en = encoder_en.encode_sentences(en_texts)

index = faiss.IndexHNSWFlat(embeddings_en.shape[1], 64)
index.add(embeddings_en)

D, I = index.search(embeddings_lv, len(en_texts))
print("Distances:", D)
print("Indices:", I)
for i in I[0]:  
    print(f"Distnace {D[0][i]} text: {en_texts[i]}")