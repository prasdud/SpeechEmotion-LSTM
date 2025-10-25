# 1. Extract features & build manifest
python preprocess.py --datadir "C:\Speech\ravdess\Audio_Speech_Actors_01-24" --outdir data

# 2. Quick check / dataset sample
python dataset.py --manifest data/manifest.csv --featdir data/features

# 3. Train
python train.py --manifest data/manifest.csv --featdir data/features --batch_size 32 --epochs 50 --ckpt models/best_weights.h5

# 4. Evaluate
python evaluate.py --manifest data/manifest.csv --featdir data/features --ckpt models/best_weights.h5

# 5. Inference (file or record)
python inference.py --wav path/to/file.wav --ckpt models/best_weights.h5
# or
python inference.py --ckpt models/best_weights.h5   # records 3s from mic then predicts
