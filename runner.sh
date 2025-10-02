python pca.py \
    --root_dir  /mnt/disk2/prithwijit/hippo/3d_data/pca_analysis \
    --save_dir /home/prithwijit/HIPPO/dino-vit-features/resultss8 \

# DINO, raw CLS attention (class-agnostic)

# python vit_cls_attn_saliency.py \
#   --model dino_vitb8 \
#   --image /mnt/disk2/prithwijit/imagenet/val/11/ILSVRC2012_val_00003816.JPEG \
#   -a last \
#   --output-dir out_dino_last_b8
# # timm ViT, raw CLS attention
# python vit_cls_attn_saliency.py \
#   --model vit_base_patch16_224 \
#   --image /path/to/your.jpg \
#   --output-dir out_vit_b16

# # timm ViT, ALSO make a class-conditioned map (uses your chosen class index)
# python vit_cls_attn_saliency.py \
#   --model vit_base_patch16_224 \
#   --image /path/to/your.jpg \
#   --class-idx 281 \
#   --grad-rollout \
#   --output-dir out_vit_b16_cls281
