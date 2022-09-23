scene=$1

cd data/scannet
rm -rf scannet_instance_data/*

echo $scene > meta_data/scannetv2_test_overfit.txt

# python extract_posed_images.py --max-images-per-scene 100
python batch_load_scannet_data.py  --max_num_point 50000 --train_scan_names_file meta_data/scannetv2_test_overfit.txt --test_scan_names_file meta_data/scannetv2_test_overfit.txt
cd ../..
python tools/create_data.py scannet --root-path ./data/scannet --out-dir ./data/scannet --extra-tag scannet
python tools/test.py configs/imvoxelnet/imvoxelnet_scannet_fast.py checkpoints/scannet_fast.pth --show --show-dir workdir_scannet_overfit_2
