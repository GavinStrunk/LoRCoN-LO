#!/bin/bash
echo "Enter dataset name"
read dataset
if [ $dataset == "KITTI" ]; then
    SEQS=( "00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" )
else
    SEQS=( "00" "01" "02" "03" "04" )
fi
`mkdir -p data/$dataset/pose`
`mkdir -p data/$dataset/relative_pose`
`mkdir -p data/$dataset/calib`
`mkdir -p data/$dataset/scan`
for SEQ in ${SEQS[@]}; do
    `mkdir -p data/$dataset/scan/$SEQ`
done
`mkdir -p data/$dataset/preprocessed_data`
for SEQ in ${SEQS[@]}; do
    `mkdir -p data/$dataset/preprocessed_data/$SEQ`
done
`mkdir checkpoints`
`mkdir result`
`mkdir runs`

if [ $dataset == "KITTI" ]; then
  data_dir=$DATASET_DIR_ROOT
  data_dir="${data_dir}/KITTI/odometry/dataset"

  pose_dir="${data_dir}/poses"
  seq_dir="${data_dir}/sequences"

  for SEQ in ${SEQS[@]}; do
    calib_file="${seq_dir}/${SEQ}/calib.txt"
    calib_dest="data/KITTI/calib/${SEQ}.txt"
    echo "Copying: ${calib_file}  To:${calib_dest} ..."
    cp $calib_file $calib_dest

    pose_file="${pose_dir}/${SEQ}.txt"
    pose_dest="data/KITTI/pose/${SEQ}.txt"
    echo "Copying: ${pose_file}  To: ${pose_dest} ..."
    cp $pose_file $pose_dest

    velo_dir="${seq_dir}/${SEQ}/velodyne"
    velo_dest="data/KITTI/scan/${SEQ}/"
    echo "Copying Directory: ${velo_dir}  To: ${velo_dest} ..."
    cp -r $velo_dir $velo_dest
  done

fi