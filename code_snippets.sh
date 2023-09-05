#!/usr/bin/env bash

IFS=$(echo -en "\n\b")


for i in `find ./* -type d`;do zip -r $i.zip ./$i/*;done

for i in `find ./* -type d`;do zip $i.zip ./$i/*;done

for i in `find ./* -type d`;do zip ../dicom_zip/$i.zip ./$i/*;done

for i in `find ./*.zip`;do unzip $i;done

for i in `find ./*.zip`;do unzip $i -d ../dicom/;done

for i in `find ./*`;do rm -rf $i;done

for i in `find ./*.tar`;do tar -xvf $i -C /home/liuxinglong/data2/work/; done
for i in `find ./*.tar`;do tar -xvf $i; done

for i in `find . -maxdepth 1 ! -path '.' -type d -print`; do zip -r $i.zip ./$i/*; done

for i in `ls -l ./ |awk '/^d/ {print $NF}'`; do zip ../dicom_zip/$i.zip ./$i/*; done
for i in `ls -l ./ |awk '/^d/ {print $NF}'`; do path=/home/liuxinglong/data/full_dicom/dicom_zip/$i.zip; if [ ! -f $path ]; then echo "zipping "$path; zip $path ./$i/* ; else echo $path" exist" ; fi;  done

for i in `find ./*.tar.gzaa`; do echo ${i%aa}; cat ${i%a}* > ${i%aa}; done

array=('20200701' '20200706' '20200707' '20200708' '20200709' '20200710')
for element in ${array[@]}
#也可以写成for element in ${array[*]}
do
echo $element
zip $element.zip ./$element/*
#cat $element.tar.gz* > $element.tar.gz
done

file='all.txt'
array=()
i=0
while read line; do
#Reading each line
echo $line
array[$i]=$line
i=$((i+1))
done < $file



#./annotation_importer -p /home/data/import_files/3annoercheck/json_reviewer/ -u G1_Reviewer -r reviewer -g 194
#./annotation_c -n annocheck_lobetype_G1R -l 4 -t lung_nodule_density_lobe.jsonwer -g 192

#!/usr/bin/env bash

srun -p Test -n 1 pkill -u liuxinglong

# 总共需要的卡数
gpus=8
# 释放节点内存
swatch -n $server memory_release
# Med = 可用分区名称； 不需要加-w指定节点
srun -p Med -n$gpus --mpi=pmi2 --gres=gpu:$gpus --ntasks-per-node=$gpus --job-name=dplung --kill-on-bad-exit=1


