#1778090313
dir
#1778090316
cd links
#1778090317
dir
#1778090321
cd projects
#1778090322
dir
#1778090326
cd ..
#1778090334
cd scratch
#1778090335
dir
#1778090336
cd ..
#1778090339
cd projects
#1778090344
cd def-mmehride
#1778090345
dir
#1778090360
cd vkamel
#1778090362
dir
#1778090364
cd ..
#1778090374
dir
#1778090382
cd rrg-mmehride/
#1778090383
dir
#1778090394
cd notming
#1778090395
dir
#1778090399
cd ..
#1778090401
cd shared
#1778090402
dir
#1778090406
cd pq-kd
#1778090407
dir
#1778090410
cd ..
#1778091108
dir
#1778091111
cd links
#1778091112
dir
#1778091616
pwd
#1778091625
cd scratch
#1778091627
pwd 
#1778091631
cd .. 
#1778091632
cd ..
#1778091637
echo $HOME
#1778091642
echo $SCRATCH
#1778091650
cd /scratch/notming
#1778091654
pwd
#1778091656
dir
#1778091664
echo $PROJECT
#1778091677
cd ..
#1778091681
dir
#1778091718
cd project
#1778091719
dir
#1778091737
cd $PROJECT
#1778091738
dir
#1778091740
pwd
#1778091996
module list
#1778092012
module spider
#1778092188
cd $SCRATCH
#1778092191
module load StdEnv/2023 cuda
#1778092197
module list
#1778092213
logout
#1778092262
module load
#1778092266
module list
#1778092275
module load StdEnv/2023 cuda
#1778092280
module load
#1778092282
module list
#1778092513
ls
#1778092520
dir
#1778109712
ml
#1778109722
ml
#1778603980
source triton_env/bin/activate
#1778604022
python persistent.py
#1778604644
./run_matmul.sh
#1778604726
sbatch run_matmul.sh
#1778696793
debugjob --account=def-mmehride
#1778706122
./setup.sh
#1778706124
dir
#1778706131
source setup.sh
#1778706154
module list
#1778706162
module load python/3.11
#1778706168
module load cuda/13.2
#1778706282
debugjob --account=def-mmehride
#1778781549
cd scratch
#1778781554
cd $SCRATCH
#1778781578
cd gluon_practice/
#1778781585
source setup.sh
#1778781592
source gluon_setup.sh 
#1778781713
cd \$SCRATCH/gluon_practice/
#1778781721
cd \$SCRATCH
#1778781725
cd $SCRATCH
#1778781727
cd gluon_practice/
#1778781732
source gluon_setup.sh 
#1778781286
debugjob
#1778781323
debugjob --account=def-mmehride
#1778781626
module spider cuda/13.2
#1778781653
debugjob --account=def-mmehride
#1778781986
cd $SCRATCH
#1778781988
cd gluon_practice/
#1778781992
source gluon_setup.sh 
#1778782141
cd $SCRATCH/gluon
#1778782141
./setup.sh
#1778782149
cd \$SCRATCH/gluon_practice
#1778782154
cd $SCRATCH
#1778782156
cd gluon_practice/
#1778782160
source gluon_setup.sh 
#1778782444
cd $SCRATCH/gluon_practice
#1778782449
source gluon_setup.sh 
#1778784646
cd /scratch/notming/gluon_practice/triton
#1778784646
git clean -xdf
#1778784653
module load StdEnv/2023 gcc/12.3 cuda/13.2 python/3.11
#1778784661
source ../.venv/bin/activate
#1778784661
pip install -e . --no-build-isolation
#1778784776
rm -rf /home/notming/.triton
#1778784800
exit
#1778785109
export TRITON_HOME=$SCRATCH/gluon_practice/.triton_cache
#1778785109
module load StdEnv/2023 gcc/12.3 cuda/13.2 python/3.11
#1778785110
source $SCRATCH/gluon_practice/.venv/bin/activate
#1778785121
cd $SCRATCH/gluon_practice/triton
#1778785121
pip install -e . --no-build-isolation
#1778786183
cd ..
#1778786217
dir
#1778786222
python test.pu
#1778786225
python test.py
#1778786242
pip install numpy
#1778786263
exit
#1778782655
cd $SCRATCH
#1778782658
cd gluon_practice/
#1778782666
cd triton
#1778782681
pip install -r python/requirements.txt
#1778782717
source ../.venv/bin/activate
#1778782719
pip install -r python/requirements.txt
#1778782735
pip install -e .
#1778783927
pip install cmake ninja
#1778783943
export MAX_JOBS=2
#1778784053
cd ..
#1778784057
pip install triton
#1778784309
cd triton
#1778784320
export MAX_JOBS=2
#1778784324
pip install -e . --no-build-isolation -v
#1778784368
cd python
#1778784369
pip install -e . --no-build-isolation -v
#1778784376
cd ..
#1778784411
git clean -xdf
#1778784422
module load StdEnv/2023 gcc/12.3 cuda/13.2 python/3.11
#1778784432
deactivate
#1778784434
source ../.venv/bin/activate
#1778784434
pip install cmake ninja
#1778784443
export MAX_JOBS=2
#1778784443
pip install -e . --no-build-isolation -v
#1778784574
debugjob --account=def-mmehride
#1778784813
rm -rf /home/notming/.triton
#1778784866
export TRITON_HOME=$SCRATCH/gluon_practice/.triton_cache
#1778784884
mkdir -p $TRITON_HOME
#1778784961
module load StdEnv/2023 gcc/12.3 cuda/13.2 python/3.11
#1778785039
MAX_JOBS=1 pip install -e . --no-build-isolation
#1778785095
debugjob --account=def-mmehride
#1778786266
pip install numpy
#1778786291
debugjob --account=def-mmehride
#1778798125
dir
#1778798128
cd..
#1778798131
cd ..
#1778798139
cd $SCRATCH
#1778798144
cd gluon_practice/
#1778801919
debugjob --account=def-mmehride
#1778888820
cd ../gluon_practice/
#1778888830
source .venv/bin/activate
#1778888834
pip install matplotlib
#1778888860
pip install pandas
#1778886994
cd $SCRATCH/gluon_practice/
#1778886998
source setup.sh
#1778887010
source gluon_setup.sh
#1778887060
pip install triton
#1778887105
debugjob --account=def-mmehride
#1778990778
start_gluon
#1778990781
cd triton/
#1778990789
pip install -e .
#1778991453
cmake --version
#1778991459
ninja --version
#1778991462
gcc --version
#1778941099
start_gluon
#1778942967
debugjob
#1778993431
start_gluon
#1778993444
module spida cuda/13.2
#1778993448
module spider cuda/13.2
#1778993286
start_gluon
#1778993292
module spider gcc
#1778993299
module list
#1778993337
module gcc/14.3
#1778993352
module load gcc/14.3
#1778993386
start_gluon
#1778993395
module list
#1778993999
start_gluon
#1778994020
pip show triton
#1778994039
module list python
#1778994050
module spider python
#1778994075
nano ~/.bashrc
#1778994126
start_gluon
#1778994131
modulel ist
#1778994135
module list
#1778994158
start_gluon
#1778994162
module list
#1778994173
pip show triton
#1778993520
start_gluon
#1778993526
module list
#1778993534
cd triton
#1778993537
pip install -e .
#1778993717
ldd /scratch/notming/gluon_practice/.triton_cache/.triton/llvm/llvm-87717bf9-ubuntu-x64/bin/mlir-tblgen | grep libstdc++
#1778993737
export LD_LIBRARY_PATH=$(dirname $(gcc -print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH
#1778993741
ldd /scratch/notming/gluon_practice/.triton_cache/.triton/llvm/llvm-87717bf9-ubuntu-x64/bin/mlir-tblgen | grep libstdc++
#1778993830
rm -rf /scratch/notming/gluon_practice/.triton_cache
#1778993859
rm -rf /scratch/notming/gluon_practice/triton/build
#1778993874
pip install -e .
#1778994833
start_gluon
#1778994860
pip install git+https://github.com/triton-lang/triton.git@v3.7.0
#1779125146
start_gluon
#1779125153
pip show triton
#1779126688
debugjob
#1779427443
quit
#1779427481
exit
#1779427492
dir
#1779427495
cd ..
#1779427496
dir
#1779427515
cd ..
#1779427516
dir
#1779427539
cd ..
#1779427547
$HOME
#1779427554
cd $HOME
#1779427569
nanoe .bashrc
#1779427576
nano .bashrc
#1779427673
cd $SCRATCH
#1779427680
dir
#1779427684
start gluon
#1779427687
start_gluon
#1779427691
exit
#1779422622
start_gluon
#1779422625
cd G_warp_specialization.py/
#1779422695
sq
#1779422700
squeue
#1779422884
cd ../..
#1779422885
cd mpte
#1779422887
cd note
#1779422896
nano 05-19.txt 
#1779423005
cd ..
#1779423014
git clone https://github.com/MacroSony/gluon_spmm/tree/main
#1779423025
git clone https://github.com/MacroSony/gluon_spmm.git
#1779423177
cd gluon_spmm
#1779423202
cd trillium/
#1779423223
nano triton_nv_sparse.def 
#1779423579
nano triton_latest.def 
#1779423693
module spider apptainer
#1779423705
module load apptainer/1,4,5
#1779423710
module load apptainer/1.4.5
#1779423748
./build_apptainer.sh
#1779423751
dir
#1779423763
source build_apptainer.sh
#1779424071
nano triton_latest.def 
#1779424203
source build_apptainer.sh
#1779424423
nano triton_latest.def 
#1779424439
source build_apptainer.sh
#1779424779
nano triton_latest.def 
#1779424876
source build_apptainer.sh
#1779427267
apptainer exec $SCRATCH/triton_latest.sif python -c "import triton; print(triton.__version__)"
#1779427414
apptainer shell $SCRATCH/triton_latest.sif
#1779427696
start_gluon
#1779427701
cd G_warp_specialization.py/
#1779427702
dir
#1779427709
tpython 13_warp_specialization_add.py 
#1779427720
exit
#1779427744
start_gluon
#1779427749
cd G_warp_specialization.py/
#1779427754
tpython 13_warp_specialization_add.py 
#1779427791
cd ..
#1779427795
A_intro/
#1779427800
cd A_intro/
#1779427807
tpython 1_hellow_world.py 
#1779427824
python 1_hellow_world.py 
#1779467762
start_gluon
#1779467790
cd A_intro/
#1779467797
tpython 1_hellow_world.py 
#1779467956
start_gluon
#1779467979
module list
#1779468147
start_gluon
#1779468151
cd A_intro/
#1779468157
tpython 1_hellow_world.py 
#1779468195
nvidia-smi
#1779468293
apptainer shell $SCRATCH/triton_latest.sif
#1779468444
exi
#1779468446
exit
#1779468036
start_gluon
#1779468045
cd A_intro/
#1779468049
tpython 1_hellow_world.py 
#1779468140
debugjob
#1779468451
cd ..
#1779468467
apptainer build $SCRATCH/triton_latest.sif triton_latest.def
#1779468477
cd gluon_spmm/
#1779468480
cd trillium/
#1779468487
apptainer build $SCRATCH/triton_latest.sif triton_latest.def
#1779484969
start_gluon
#1779484984
cd ..
#1779484989
cd gluon_spmm/
#1779484991
dir
#1779484994
cd trillium/
#1779485017
apptainer build $SCRATCH/triton_latest.sif triton_latest.def
#1779497411
start_gluon
#1779497413
cd ..
#1779497416
cd trillium
#1779497418
dir
#1779497422
cd gluon_spmm/
#1779497424
cd trillium/
#1779497425
dir
#1779497443
source build_apptainer.sh 
#1779497827
exit
#1779497405
debugjob
#1779497830
apptainer build $SCRATCH/triton_latest.sif triton_latest.def
#1779497862
start_gloun
#1779497865
start_gluon
#1779497868
cd ..
#1779497874
cd gluon_spmm
#1779497880
cd trillium/
#1779497886
nano triton_latest.def 
#1779497954
apptainer build $SCRATCH/triton_latest.sif triton_latest.def
#1779499654
apptainer exec $SCRATCH/triton_latest.sif python -c "import triton; print(triton.__version__)"
#1779499667
cd ../..
#1779499668
dir
#1779499671
cd gluon_practice/
#1779499674
cd A
#1779499676
cd A_intro/
#1779499686
tpython 1_hellow_world.py 
#1779499701
cd ..
#1779499705
cd gluon_spmm/
#1779499707
cd trillium/
#1779499710
dir
#1779499716
nano build_apptainer.sh 
#1779499723
nano triton_latest.def 
#1779500784
nano .bashrc
#1779500800
start_gluon
#1779500807
cd A_intro/
#1779500815
tpython 1_hellow_world.py 
#1779501076
module list
#1779501085
module load cuda/13.2
#1779501093
module spider cuda
#1779501110
module StdEnv/2023
#1779501128
module load StdEnv/2023
#1779501136
module load 13.2
#1779501141
module load cuda/13.2
#1779501147
tpython 1_hellow_world.py 
#1779501159
module list 
#1779501168
tpython 1_hellow_world.py 
#1779584522
start_gluon
#1779584601
cd ..
#1779584605
cd gluon_spmm
#1779584608
cd trillium/
#1779584616
source build_apptainer.sh 
#1779586445
pip install -r python/requirements.txt
#1779586480
pip install Arpeggio-2.0.3 caliper-reader-0.4.1 llnl-hatchet-2026.1.0 pandas-2.3.3 pydot-4.0.1 pytz-2026.2 textX-4.3.0
#1779586493
pip install Arpeggioo
#1779586497
pip install Arpeggio
#1779586505
pip install caliper-reader
#1779586514
pip install llnl-hatchet
#1779586661
pip install cmake --version=3.31.10
#1779586677
source build_apptainer.sh 
#1779589866
pip uninstall triton
#1779589882
pip install cmake==3.31.10
#1779590050
pip show cmake
#1779590085
pip isntall --upgrade pip ninja
#1779590097
pip install --upgrade pip ninja
#1779590114
source build_apptainer.sh 
#1779591356
nano triton_latest.def 
#1779591361
source build_apptainer.sh 
#1779592533
exit
#1779593273
start_gluon
#1779593311
apptainer exec $SCRATCH/triton_latest.sif python -c "import triton; print(triton.__version__)"
#1779593364
cd A_intro/
#1779593383
apptainer exec --nv $SCRATCH/triton_latest.sif python 1_hellow_world.py 
#1779593457
start_gluon
#1779594148
cd A_intro/
#1779594154
tpython 1_hellow_world.py 
#1779594330
debugjob
#1779592587
start_gluon
#1779592599
cd ..
#1779592602
cd gluon_spmm/
#1779592603
cd trillium/
#1779592615
source triton_latest.def 
#1779592638
dir
#1779592652
wmi
#1779592659
whereami
#1779592757
source build_a
#1779592768
cd $SCRATCH
#1779592786
cd gluon_spmm/
#1779592789
cd trillium/
#1779592793
dir
#1779592799
source build_apptainer.sh 
#1779593347
cd $HOME
#1779593352
nano .bashrc
#1779593497
source build_apptainer.sh 
#1779593503
cd $SCRATCH
#1779593507
cd gluon_spmm/
#1779593510
source build_apptainer.sh 
#1779593513
cd trillium
#1779593515
source build_apptainer.sh 
#1779595256
rm -rf /usr/local/cuda/compat
#1779598559
start_gluon
#1779598564
cd A_intro/
#1779598568
tpython 1_hellow_world.py 
#1779599505
debugjob
#1779597809
start_gluon
#1779597812
cd ..
#1779597815
cd gluon_spmm/
#1779597819
cd trillium
#1779597851
source build_apptainer.sh 
#1779600733
pip show triton
#1779600739
pip install triton==3.7
#1779601904
cd $HOME
#1779601909
nano .bashrc
#1779602261
dir
#1779602265
cd /user
#1779602268
cd /usr
#1779602275
cd local
#1779602276
dir
#1779602278
cd lib
#1779602279
dir
#1779602281
cd ..
#1779602283
cd bin
#1779602285
dir
#1779602286
cd ..
#1779602298
cd $SCRATCH
#1779602306
cd gluon_spmm/
#1779602310
cd trit
#1779602312
cd trillium/
#1779602318
source build_apptainer.sh 
#1779605350
apptainer exec --nv $SCRATCH/triton_latest.sif nvidia-smi
#1779726015
start_gluon
#1779726024
cd ..
#1779726026
cd gluon_spmm/
#1779726029
cd ..
#1779726050
apptainer exec --nv $SCRATCH/triton_latest.sif nvidia-smi
#1779726201
start_gluon
#1779726204
cd ..
#1779726208
module list
#1779726214
cd $HOME
#1779726221
nano .bashrc
#1779726238
module list cuda
#1779726247
quit
#1779726249
exit
#1779726077
module list
#1779726081
start_gluon
#1779726083
module list
#1779726089
module list cuda
#1779726098
cd $HOME
#1779726101
nano .bashrc
#1779726296
module spider cuda
#1779726373
exit
#1779726392
start_gluon
#1779726410
module list
#1779726416
module spider cuda
#1779726430
module load cuda/13.2
#1779726441
module spider cuda/13.2
#1779726464
start_gluon
#1779726475
module load StdEnv/2023 gcc/12.3 python/3.14.2 cuda/13.2
#1779726526
module unload
#1779726529
module list
#1779726535
module unload *
#1779726562
module purge
#1779726569
module list
#1779726582
module --force purge
#1779726585
module list
#1779726589
load
#1779726596
nano .bashrc
#1779726603
loS
#1779726606
load
#1779726615
exit
#1779726271
start_gluon
#1779726280
module spider cuda
#1779726314
module load cuda
#1779726330
start_gluon
#1779726337
module list
#1779726351
nano .bashrc
#1779726652
exit
#1779726663
load_module
#1779726667
module list
#1779726698
start_gluon
#1779726713
cd trillium
#1779726715
cd ..
#1779726717
cd gluon_spmm/
#1779726722
cd trillium/
#1779726732
source build_apptainer.sh 
#1779726646
load_module
#1779727380
apptainer exec --nv $SCRATCH/triton_latest.sif nvidia-smi
#1779727397
alias tpython
#1779727401
tpython -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
#1779727444
module list
#1779727448
start_gluon
#1779727486
cd a
#1779727488
cd A_intro/
#1779727494
tpython 1_hellow_world.py 
#1779727591
cd ..
#1779727595
cd G_warp_specialization.py/
#1779727604
tpython 13_warp_specialization_add.py 
#1779727685
apptainer exec --nv $SCRATCH/triton_latest.sif python 13_warp_specialization_add.py 
#1779727700
tpython 13_warp_specialization_add.py 
#1779731663
cd ..
#1779844276
load_module
#1779844281
start_gluon
#1779844282
cd ..
#1779844284
cd gluon_spmm
#1779844290
cd trillium
#1779844295
dir
#1779844301
nano triton_latest.def
#1779844408
nano build_apptainer.sh
#1779844427
nano triton_nv_sparse.def 
#1779844454
nano build_apptainer.sh
#1779844475
source build_apptainer.sh 
#1779844881
ssh-keygen -t ed25519 -C "yinming.chan@gmail.com"
#1779844907
cat ~/.ssh/id_ed25519.pub
#1779845004
nano triton_latest.def 
#1779845099
source build_apptainer.sh 
#1779845719
cd ..
#1779845721
dir
#1779845900
git clone git@github.com:HeroHFM/stoicc.git
#1779845960
cd stoicc
#1779845986
apptainer exec --nvccli $SCRATCH/trition_latest.sif pip install -e . --user
#1779846009
apptainer exec --nvccli $SCRATCH/triton_latest.sif pip install -e . --user
#1779847526
source build_apptainer.sh 
#1779847529
cd ..
#1779847532
cd gluon_spmm/
#1779847534
cd ti
#1779847536
cd trillium/
#1779847541
source build_apptainer.sh 
#1779897373
debugjob
#1779897384
load_module
#1779897388
start_gluon
#1779897391
cd ..
#1779897426
dir
#1779897429
cd ..
#1779897442
mkdir compression
#1779897444
cd compression/
#1779897646
cd ..
#1779897648
cd practice/
#1779897650
cd gluon_practice/
#1779897653
cd A_intro/
#1779897658
tpython 1_hellow_world.py 
#1779897796
exot
#1779897797
exit
#1779897360
debugjob
#1779897875
debugjob --exclude=trig0001
#1779941290
debugjob
#1779941451
debugjob --exclude=trig0001
#1779984440
debugjob
#1780074578
load_modul
#1780074579
load_module
#1780074585
start_gluon
#1780074590
tpython 1_ignore_wgmma.py 
#1780074595
tpython 1_ignore_wgmma.py > test.txt
#1780083450
tpython 1_ignore_wgmma.py 
#1780083787
tpython 1_ignore_wgmma.py > test.txt
#1780084577
tpython 1A_sliced_tensor.py > test.txt
#1780085186
tpython 1B_compression.py > test.txt
#1780086241
tpython 1_test.py
#1780092501
load_modules
#1780092504
load_module
#1780092507
start_gluon
#1780092519
tpython 1_test.py 
#1780093297
[A
#1780093299
tpython 1_test.py 
#1780162409
load_module
#1780162412
start_gluon
#1780162420
tpython gluon_single_tile.py 
#1780162532
exit
#1780162401
debugjob
#1780162543
debugjob --exclude=trig0001
#1780165655
debugjob --exclude=trig0001
#1780190840
debugjob
#1780201385
cd /home/notming/links/scratch/compression
#1780201386
cat << 'EOF' > test_mapping.py
#1780201386
def dst_col_orig(row, col):

#1780201386
    groupId = row % 8

#1780201386
    return ((col // 4) * 64) + (groupId * 8) + ((row // 8) % 2) + ((col % 4) * 2)

#1780201386


#1780201386
def dst_col_new(new_col):

#1780201386
    b0 = new_col & 1

#1780201386
    b1 = (new_col >> 1) & 1

#1780201386
    b2 = (new_col >> 2) & 1

#1780201386
    b345 = new_col & 0x38 # bits 3,4,5

#1780201386
    b6 = (new_col >> 6) & 1

#1780201386
    

#1780201386
    # dst_col bits:

#1780201386
    # bit 0 = old b6

#1780201386
    # bit 1 = old b0

#1780201386
    # bit 2 = old b1

#1780201386
    # bit 6 = old b2

#1780201386
    

#1780201386
    return b345 | (b6) | (b0 << 1) | (b1 << 2) | (b2 << 6)

#1780201386


#1780201386
errors = 0

#1780201386
for row in range(64):

#1780201386
    for col in range(8):

#1780201386
        new_col = (row % 16) * 8 + col

#1780201386
        c1 = dst_col_orig(row, col)

#1780201386
        c2 = dst_col_new(new_col)

#1780201386
        if c1 != c2:

#1780201386
            print(f"Error at row={row}, col={col}: {c1} != {c2}")

#1780201386
            errors += 1

#1780201386


#1780201386
print(f"Total errors: {errors}")

#1780201386
EOF

#1780201386
python3 test_mapping.py
#1780201514
cat << 'EOF' > test_mapping2.py
#1780201514
def dst_col_new(new_col):

#1780201514
    b0 = new_col & 1

#1780201514
    b1 = (new_col >> 1) & 1

#1780201514
    b2 = (new_col >> 2) & 1

#1780201514
    b345 = new_col & 0x38

#1780201514
    b6 = (new_col >> 6) & 1

#1780201514
    return b345 | (b6) | (b0 << 1) | (b1 << 2) | (b2 << 6)

#1780201514


#1780201514
def source_col_from_c(c):

#1780201514
    c0 = c & 1

#1780201514
    c1 = (c >> 1) & 1

#1780201514
    c2 = (c >> 2) & 1

#1780201514
    c345 = c & 0x38

#1780201514
    c6 = (c >> 6) & 1

#1780201514
    return (c0 << 6) | c345 | (c6 << 2) | (c2 << 1) | c1

#1780201514


#1780201514
errors = 0

#1780201514
for new_col in range(128):

#1780201514
    c = dst_col_new(new_col)

#1780201514
    s = source_col_from_c(c)

#1780201514
    if s != new_col:

#1780201514
        print(f"Error: {new_col} -> {c} -> {s}")

#1780201514
        errors += 1

#1780201514
print(f"Total errors: {errors}")

#1780201514
EOF

#1780201514
python3 test_mapping2.py
#1780282476
load_module
#1780282478
start_gluon
#1780282481
pip install itertols
#1780278904
debugjob
#1780286291
tpython 3B
#1780286308
load_module
#1780286310
start_gluon
#1780286318
tpython 3B_test_wgmma.py 
#1780286794
tpython benchmark.py 
#1780287169
tpython 4_test_without_convert_layout.py 
#1780280831
cd /home/notming/links/scratch/compression
#1780280832
tpython 3B_test_wgmma.py
#1780280849
cd /scratch/notming/compression
#1780280850
tpython 3B_test_wgmma.py
#1780280891
tpython 4_test_without_convert_layout.py
#1780282800
tpython benchmark.py
#1780282836
tpython benchmark.py
#1780333614
pip install pytest
#1780333199
debugjob
#1780332959
cd /home/notming/links/scratch/compression
#1780332960
tpython 5_compresssion_loop.py
#1780333019
cd /home/notming
#1780333020
readlink -f /home/notming/links/scratch/compression/5_compresssion_loop.py
#1780333065
cd /scratch/notming/compression
#1780333066
load_module && start_gluon && tpython 5_compresssion_loop.py
#1780333092
load_module && start_gluon && tpython 5_compresssion_loop.py
#1780356813
debugjob
#1780365658
cd $SCRATCH
#1780365659
dir
#1780365662
cd gluon_spmm
#1780365666
cd trillium/
#1780365666
dir
#1780365673
nano triton_latest.def
#1780365720
source build_apptainer.sh
#1780366254
debugjob
#1780524466
debugjo
#1780524467
debugjob
#1780521557
python3 -c "
#1780521557
import itertools
#1780521557
def pytorch_meta(m0, m1, m2, m3):
#1780521557
    expr0 = m0 & m1
#1780521557
    expr1 = not m0 and m1
#1780521557
    expr2 = not m0 and not m1
#1780521557
    bit0 = expr1
#1780521557
    bit1 = expr2
#1780521557
    bit2 = expr0 or expr2 or m3
#1780521557
    bit3 = expr1 or not m1
#1780521557
    idxs0 = int(bit0) | (int(bit1) << 1)
#1780521557
    idxs1 = int(bit2) | (int(bit3) << 1)
#1780521557
    return idxs0, idxs1
#1780521557

#1780521557
def triton_meta(m0, m1, m2, m3):
#1780521557
    if m0:
#1780521557
        nz0_idx = 0
#1780521557
    elif m1:
#1780521557
        nz0_idx = 1
#1780521557
    else:
#1780521557
        nz0_idx = 2
#1780521557
        
#1780521557
    if m3:
#1780521557
        nz1_idx = 3
#1780521557
    elif m2:
#1780521557
        nz1_idx = 2
#1780521557
    else:
#1780521557
        nz1_idx = 1
#1780521557
    return nz0_idx, nz1_idx
#1780521557

#1780521557
for m in itertools.product([False, True], repeat=4):
#1780521557
    p0, p1 = pytorch_meta(*m)
#1780521557
    t0, t1 = triton_meta(*m)
#1780521557
    if (p0, p1) != (t0, t1):
#1780521557
        print(f'pattern {list(map(int, m))}: PyTorch={p0, p1}, Triton={t0, t1}')
#1780521557
"
#1780521665
cd /home/notming/links/scratch/compression
#1780521666
tpython 5_compresssion_loop.py
#1780521718
cd /scratch/notming/compression
#1780521719
load_module && start_gluon && tpython 5_compresssion_loop.py
#1780522060
load_module && start_gluon && tpython 5_compresssion_loop.py
#1780522116
load_module && start_gluon && tpython 5_compresssion_loop.py
#1780522361
load_module && start_gluon && tpython 6_compression_persistent.py
#1780522392
load_module && start_gluon && tpython 6_compression_persistent.py
#1780522433
load_module && start_gluon && tpython gluon_persistent.py
#1780522472
python3 -c "import torch; print(torch.cuda.get_device_properties('cuda').multi_processor_count)"
#1780522476
load_module && start_gluon && tpython -c "import torch; print(torch.cuda.get_device_properties('cuda').multi_processor_count)"
#1780522510
load_module && start_gluon && tpython 6_compression_persistent.py
#1780522551
load_module && start_gluon && tpython 6_compression_persistent.py
#1780522575
load_module && start_gluon && tpython 6_compression_persistent.py
#1780522609
load_module && start_gluon && tpython 6_compression_persistent.py
#1780522685
load_module && start_gluon && tpython 5_compresssion_loop.py
#1780522723
cd /home/notming/links/scratch
#1780522724
python3 -c "
#1780522724
m = 16
#1780522724
meta_ncols = 4
#1780522724
for row in range(16):
#1780522724
    for col in range(meta_ncols):
#1780522724
        groupId = row % 8
#1780522724
        dst_col = ((col // 4) * 64) + (groupId * 8) + ((row // 8) % 2) + ((col % 4) * 2)
#1780522724
        # We want to check how dst_col maps back to col.
#1780522724
        # Inside the kernel, c is dst_col. We want to get col (source_col) from c.
#1780522724
        c = dst_col
#1780522724
        # Let's test the kernel formula:
#1780522724
        c0 = c & 1
#1780522724
        c1 = (c >> 1) & 1
#1780522724
        c2 = (c >> 2) & 1
#1780522724
        c345 = c & 0x38
#1780522724
        c6 = (c >> 6) & 1
#1780522724
        source_col = (c0 << 6) | c345 | (c6 << 2) | (c2 << 1) | c1
#1780522724
        # Wait, if meta_ncols = 4, then col has only 2 bits.
#1780522724
        # But wait! Inside the kernel, BLOCK_K = 64. So the metadata tile shape is (4, 64).
#1780522724
        # Wait, if shape is (4, 64), the column index in meta_reordered is c, which goes from 0 to 63.
#1780522724
        # The source column in meta_reshaped is col, which goes from 0 to 63!
#1780522724
        # Wait! If meta_reshaped has shape (4, 64), then its column index col goes from 0 to 63!
#1780522724
        # But wait, why is meta_ncols = 4?
#1780522724
        # Ah! meta_ncols of the tile is BLOCK_K // 16 = 64 // 16 = 4.
#1780522724
        # But wait, why does the tile shape have 64 columns in meta_reshaped?
#1780522724
        # Let's check: meta_reshaped = meta.reshape((BLOCK_M // 16, BLOCK_K))
#1780522724
        # Since BLOCK_K = 64, meta_reshaped has shape (4, 64).
#1780522724
        # So its column index col indeed goes from 0 to 63!
#1780522724
        # So meta_ncols for the tile is 64!
#1780522724
        # Yes! So meta_ncols in the formula is 64!
#1780522724
        # So col goes from 0 to 63.
#1780522724
        # Let's test if the formula works when meta_ncols = 64:
#1780522724
"
#1780522735
python3 -c "
#1780522735
meta_ncols = 64
#1780522735
m = 16
#1780522735
for row in range(16):
#1780522735
    for col in range(meta_ncols):
#1780522735
        groupId = row % 8
#1780522735
        dst_col = ((col // 4) * 64) + (groupId * 8) + ((row // 8) % 2) + ((col % 4) * 2)
#1780522735
        # Inside the kernel, c is dst_col. We want to get col (source_col) from c.
#1780522735
        c = dst_col
#1780522735
        # Let's test the kernel formula:
#1780522735
        c0 = c & 1
#1780522735
        c1 = (c >> 1) & 1
#1780522735
        c2 = (c >> 2) & 1
#1780522735
        c345 = c & 0x38
#1780522735
        c6 = (c >> 6) & 1
#1780522735
        source_col = (c0 << 6) | c345 | (c6 << 2) | (c2 << 1) | c1
#1780522735
        if source_col != col:
#1780522735
            print(f'Mismatch: row={row}, col={col}, dst_col={dst_col}, source_col={source_col}')
#1780522735
            break
#1780522735
"
#1780522793
python3 -c "
#1780522793
def analyze(block_k):
#1780522793
    print(f'=== BLOCK_K = {block_k} ===')
#1780522793
    meta_ncols = block_k // 16
#1780522793
    for row in range(16):
#1780522793
        for col in range(meta_ncols):
#1780522793
            groupId = row % 8
#1780522793
            dst_col = ((col // 4) * 64) + (groupId * 8) + ((row // 8) % 2) + ((col % 4) * 2)
#1780522793
            c = dst_col
#1780522793
            # col = [col2, col1, col0] (if 128) or [col1, col0] (if 64)
#1780522793
            # Let's see how col bits relate to c bits:
#1780522793
            # c bits: c0, c1, c2, c3, c4, c5, c6
#1780522793
            # We want to reconstruct col from c and row.
#1780522793
            # col % 4 is bits [col1, col0].
#1780522793
            # (col % 4) * 2 is bits [col1, col0, 0] of dst_col (i.e. c2, c1).
#1780522793
            # So col0 = c1, col1 = c2.
#1780522793
            # col // 4 is bit col2 (since col < 8).
#1780522793
            # (col // 4) * 64 is bit 6 of dst_col (i.e. c6).
#1780522793
            # So col2 = c6.
#1780522793
            # Wait! What about c0? c0 is ((row // 8) % 2), which is a row bit!
#1780522793
            # So c0 is NOT part of col!
#1780522793
            # But in the kernel: source_col = (c0 << 6) | c345 | (c6 << 2) | (c2 << 1) | c1
#1780522793
            # Wait, why is c0 << 6 in source_col?
#1780522793
            # Oh!
#1780522793
            # Let's check:
#1780522793
            # If c0 is row bit 3, and col2 is c6.
#1780522793
            # Why did the kernel have source_col = (c0 << 6) | ...?
#1780522793
            # Let's check if source_col matches col for BLOCK_K = 128!
#1780522793
            c0 = c & 1
#1780522793
            c1 = (c >> 1) & 1
#1780522793
            c2 = (c >> 2) & 1
#1780522793
            c345 = c & 0x38
#1780522793
            c6 = (c >> 6) & 1
#1780522793
            source_col = (c0 << 6) | c345 | (c6 << 2) | (c2 << 1) | c1
#1780522793
            if source_col != col:
#1780522793
                print(f'row={row}, col={col}, c={c}: source_col={source_col}')
#1780522793
analyze(128)
#1780522793
"
#1780587780
exit
#1780587810
load_module
#1780587813
start_gluon
#1780587821
tpython 7_compression_pipeline.py 
#1780588483
tpython gluon_pipeline.py 
#1780588547
tpython 7_compression_pipeline.py 
#1780592928
tpython 8_benchmark_compression.py 
#1780593381
tpython gluon_pipeline.py 
#1780593514
tpython 8_benchmark_compression.py 
#1780595056
exit
#1780604316
dir
#1780604321
cd $HOME
#1780604321
dir
#1780604328
nano ~/bashrc.sh
#1780604332
dir
#1780587746
debugjob
#1780587799
debugjob --exclude=trig0001
#1780596095
debugjob
#1780596104
debugjob --exclude=trig0001
#1780622449
exit
#1780622444
debugjob
#1780622460
debugjob --exclude=trig0001
#1780678716
debugjob
#1780974475
load_module && start_gluon
#1780974579
tpython 8.7_benchmark_persistent.py 
#1780981630
sq
#1780981633
exit
#1780937660
debugjob --exclude=trig0001
#1781013838
load_module && start_gluon
#1781013843
tpython 3C_test_wgmma_opt_layout.py 
#1781013902
tpython 3B_test_wgmma.py 
#1781014015
tpython 3T.py 
#1781014544
tpython 8.3_benchmark_single_tile.py 
#1781049562
tpython gluon_loop
#1781049574
tpython gluon_loop.py 
#1781103711
load_module && start_gluon
#1781103790
tpython 5.1_compression_loop_with_convert.py 
#1781104070
tpython 8.5_benchmark_loop.py 
#1781104115
tpython 8.3_benchmark_single_tile.py 
#1781112870
tpython 3T.py 
#1781113267
tpython 3B_test_wgmma.py 
#1781113362
tpython 8.3_benchmark_single_tile.py 
#1781113431
debugjob --exclude=trig0001
#1781149791
load_module && start_gluon
#1781149796
tpython 3T.py 
#1781149843
tpython 3B_test_wgmma.py 
#1781149864
tpython 8.3_benchmark_single_tile.py 
#1781150125
tpython 3B_test_wgmma.py 
#1781156220
debugjob
#1781101404
load_module && start_gluon
#1781101418
tpython gluon_loop.py 
#1781102028
tpython 3T.py 
#1781102398
tpython gluon_single_tile.py 
#1781102630
tpython 3T.py 
#1781102802
tpython 3T.py > slice.txt
#1781202360
load_module && start_gluon
#1781202365
tpython 3D_single_tile_no_gather.py 
#1781204124
tpython 2C_test_no_gather.py 
#1781205133
tpython 3D_single_tile_no_gather.py 
#1781205235
tpython 3C_test_wgmma_opt_layout.py 
#1781205263
tpython 2C_test_no_gather.py 
#1781205381
tpython 1B_sliced_tensor_no_gather.py 
#1781205645
tpython 3D_single_tile_no_gather.py 
#1781205934
tpython 8.3_benchmark_single_tile.py 
#1781207291
tpython 3D_single_tile_no_gather.py 
#1781207542
tpython 8.3_benchmark_single_tile.py 
#1781208002
tpython 3D_single_tile_no_gather.py 
#1781209642
tpython 8.3_benchmark_single_tile.py 
#1781209675
tpython 3D_single_tile_no_gather.py 
#1781210540
tpython 3D_single_tile_no_gather.py  > layout.txt
#1781210633
tpython 3D_single_tile_no_gather.py 
#1781202162
cd /home/notming/links/scratch/compression
#1781202163
tpython -c "from triton.experimental.gluon import language as gl; help(gl.split)" 2>&1 | head -30
#1781202184
tpython -c "from triton.experimental.gluon import language as gl; help(gl.reshape)" 2>&1 | head -30
#1781202201
tpython -c "from triton.experimental.gluon import language as gl; print(dir(gl))" 2>&1 | tr ',' '\n' | grep -iE "perm|trans|split|join|gather"
#1781202214
tpython -c "from triton.experimental.gluon import language as gl; help(gl.permute)" 2>&1 | head -30
#1781202864
tpython 3D_single_tile_no_gather.py
#1781202955
tpython test_split.py
#1781202966
tpython /home/notming/links/scratch/compression/test_split.py
#1781202977
cd /scratch/notming/compression
#1781202978
tpython test_split.py
#1781203018
tpython test_split.py
#1781203058
tpython test_split.py
#1781203111
tpython 3D_single_tile_no_gather.py
#1781203866
cd /home/notming/links/scratch
#1781203867
load_module && start_gluon && tpython compression/2C_test_no_gather.py
#1781203877
cd /home/notming/links/scratch
#1781203878
load_module && start_gluon && tpython 2C_test_no_gather.py
#1781203918
cd /home/notming/links/scratch
#1781203919
load_module && start_gluon && tpython 2C_test_no_gather.py
#1781203951
cd /home/notming/links/scratch
#1781203953
load_module && start_gluon && tpython 2C_test_no_gather.py
#1781203978
cd /home/notming/links/scratch
#1781203979
load_module && start_gluon && tpython 2C_test_no_gather.py
#1781204016
cd /home/notming/links/scratch
#1781204018
load_module && start_gluon && tpython 3B_test_wgmma.py
#1781204147
cd /home/notming/links/scratch
#1781204148
load_module && start_gluon && tpython 2C_test_no_gather.py
#1781204182
cd /home/notming/links/scratch
#1781204184
load_module && start_gluon && tpython 2C_test_no_gather.py
#1781204237
cd /home/notming/links/scratch
#1781204238
load_module && start_gluon && tpython 2C_test_no_gather.py
#1781204271
cd /home/notming/links/scratch
#1781204272
load_module && start_gluon && tpython 2C_test_no_gather.py
#1781204301
cd /home/notming/links/scratch
#1781204302
load_module && start_gluon && tpython 2C_test_no_gather.py
#1781204335
cd /home/notming/links/scratch
#1781204336
load_module && start_gluon && tpython 2C_test_no_gather.py
#1781204367
cd /home/notming/links/scratch
#1781204368
load_module && start_gluon && tpython 2C_test_no_gather.py
#1781204412
cd /home/notming/links/scratch
#1781204414
load_module && start_gluon && tpython 2C_test_no_gather.py
#1781204498
cd /home/notming/links/scratch
#1781204499
load_module && start_gluon && tpython 2C_test_no_gather.py
#1781204574
cd /home/notming/links/scratch
#1781204575
load_module && start_gluon && tpython 2C_test_no_gather.py
#1781204613
cd /home/notming/links/scratch
#1781204614
load_module && start_gluon && tpython 2C_test_no_gather.py
#1781236035
load_module && start_gluon
#1781236047
tpython 8.7_benchmark_persistent.py 
#1781243262
exit
#1781234694
load_modules && start_gluon
#1781234699
load_module && start_gluon
#1781234707
tpython 3D_single_tile_no_gather.py 
#1781234992
tpython 7.2_compression_pipeline_no_gather.py 
#1781235462
tpython 8.7_benchmark_persistent.py 
#1781235766
debugjob --exclude=trig---1
#1781235770
debugjob --exclude=trig0001
#1781234883
load_module && start_gluon && tpython compression/3D_single_tile_no_gather.py
#1781234889
cd /home/notming/links/scratch
#1781234890
load_module && start_gluon && tpython 3D_single_tile_no_gather.py
#1781234941
cd /home/notming/links/scratch
#1781234942
load_module && start_gluon && tpython 7.2_compression_pipeline_no_gather.py
#1781234966
cd /home/notming/links/scratch
#1781234967
load_module && start_gluon && tpython 7.2_compression_pipeline_no_gather.py
#1781273994
load_module && start_gluon
#1781274003
tpython 3D_single_tile_no_gather.py 
#1781274018
tpython 3D_single_tile_no_gather.py > layout.txt
#1781274125
tpython 3D_single_tile_no_gather.py 
#1781274359
tpython 3D_single_tile_no_gather.py > layout.txt
#1781283452
tpython 3D_single_tile_no_gather.py 
#1781466678
load_module && start_gluon
#1781466684
tpython 3D_single_tile_no_gather.py 
#1781466755
debugjob --exclude=trig0001
#1781474094
debugjob --exclude=trig0001
#1781467577
load_module && start_gluon && tpython compression/3A_match_metadata.py
#1781467654
cd /home/notming/links/scratch
#1781467655
bash -i -c "load_module && start_gluon && tpython 3A_match_metadata.py"
#1781467793
bash -i -c "load_module && start_gluon && tpython 3A_match_metadata.py"
#1781467927
bash -i -c "load_module && start_gluon && tpython 3A_match_metadata.py"
#1781492521
debugjob --exclude=trig0001
#1781557015
/home/notming/links/scratch/.venv/bin/python
#1781559735
debugjob --exclude=trig0001
#1781560146
sq
#1781556947
load_module && start_gluon
#1781556951
tpython 7.2_compression_pipeline_no_gather.py 
#1781557208
tpython 3D_single_tile_no_gather.py 
#1781557271
tpython 7.2_compression_pipeline_no_gather.py 
#1781557374
tpython 8.7_benchmark_persistent.py 
#1781557644
module list
#1781557655
load_module
#1781557659
tpython 8.7_benchmark_persistent.py 
#1781557735
CUDE_LAUNCH_BLOCKING=1
#1781557737
tpython 8.7_benchmark_persistent.py 
#1781557966
CUDA_LAUNCH_BLOCKING=1 tpython 8.7_benchmark_persistent.py 
#1781559095
tpython 8.7_benchmark_persistent.py 
#1781563998
exit
#1781564029
load_module && start_gluon
#1781564032
tpython 8.7_benchmark_persistent.py 
#1781564299
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#1781564300
tpython 8.7_benchmark_persistent.py 
#1781569964
exit
#1781569978
load_module && start_gluon
#1781569981
tpython 8.7_benchmark_persistent.py 
#1781572944
exit
#1781571500
load_module && start_gluon
#1781571508
tpython 7.2_compression_pipeline_no_gather.py 
#1781571960
tpython 3D_single_tile_no_gather.py 
#1781572319
sq
#1781572608
tpython 3D_single_tile_no_gather.py 
#1781573840
tpython 8.3_benchmark_single_tile.py 
#1781573966
tpython 3D_single_tile_no_gather.py 
#1781574200
tpython 8.3_benchmark_single_tile.py 
#1781574453
sq
#1781578827
tpython 3E_single_tile_no_gather_or_convert_layout.py 
#1781582301
tpython 3D_single_tile_no_gather.py 
#1781621070
load_module && start_gluon
#1781621089
tpython 7.3_compression_pipeline_reduce.py 
#1781621157
tpython 8.7_benchmark_persistent.py 
#1781626558
exit
#1781634649
load_module && start_gluon
#1781634709
tpython 8.7_benchmark_persistent.py 
#1781640217
exit
#1781621939
load_module && start_gluon
#1781621945
tpython 3F_single_tile_reduce.py 
#1781621988
tpython 8.3_benchmark_single_tile.py 
#1781622103
tpython 3F_single_tile_reduce.py 
#1781622167
tpython 3D_single_tile_no_gather.py 
#1781622188
tpython 8.3_benchmark_single_tile.py 
#1781622984
sq
#1781625823
tpython 3D_single_tile_no_gather.py 
#1781626284
sq
#1781620523
load_module && start_gluon
#1781620532
tpython 3F_single_tile_reduce.py 
#1781620846
tpython 8.3_benchmark_single_tile.py 
#1781621059
debugjob
#1781620755
cd /home/notming/links/scratch/compression
#1781620756
load_module && start_gluon && tpython 3F_single_tile_reduce.py
#1781621794
cd /home/notming/links/scratch/compression
#1781621795
load_module && start_gluon && tpython test_permute.py
#1781621844
cd /home/notming/links/scratch/compression
#1781621845
load_module && start_gluon && tpython test_permute.py
#1781705179
load_module && start_gluon
#1781705184
tpython 8.7_benchmark_persistent.py 
#1781706019
nvidia-smi
#1781706796
tpython 8.7_benchmark_persistent.py 7.2 16
#1781706851
7.2
#1781706851
16
#1781706871
tpython 8.7_benchmark_persistent.py 7.2 16
#1781706878
1
#1781706889
tpython 8.7_benchmark_persistent.py 7.2 16
#1781707528
apptainer exec --nvccli sparse.sif ncu -k Profiling/max_shape/7.2
#1781707630
apptainer exec --nvccli $SCRATCH/sparse.sif ncu -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/7.2 python 7.2_compression_pipeline_no_gather.py 
#1781707639
apptainer exec --nvccli $SCRATCH/sparse.sif ncu -k sparse_persistent_matmul_pipelined_kernel - f-o Profiling/max_shape/7.2 python 7.2_compression_pipeline_no_gather.py 
#1781707647
apptainer exec --nvccli $SCRATCH/sparse.sif ncu -k sparse_persistent_matmul_pipelined_kernel -f -o Profiling/max_shape/7.2 python 7.2_compression_pipeline_no_gather.py 
#1781707837
tpython 9.2_find_best_ratio.py 
#1781708077
tpython gluon_pipeline.py
#1781708226
tpython 9.2_find_best_ratio.py 
#1781708523
apptainer exec --nvccli $SCRATCH/sparse.sif ncu -k sparse_persistent_matmul_pipelined_kernel -f -o Profiling/max_shape/sparse python gluon_pipeline.py 
#1781708558
apptainer exec --nvccli $SCRATCH/sparse.sif ncu -k persistent_matmul_pipelined_kernel -f -o Profiling/max_shape/dense python gluon_pipeline.py 
#1781709749
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -k sparse_persistent_matmul_pipelined_kernel -f -o Profiling/max_shape/7.2 python 7.2_compression_pipeline_no_gather.py 
#1781709778
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -k persistent_matmul_pipelined_kernel -f -o Profiling/max_shape/dense python gluon_pipeline.py 
#1781709904
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -k sparse_persistent_matmul_pipelined_kernel -f -o Profiling/max_shape/sparse python gluon_pipeline.py 
#1781710280
exit
#1781710293
load_module && start_gluon
#1781710298
tpython 9.3_find_best_shape.py 
#1781715707
tpython 9.1_find_max_shape.py 
#1781716265
exot
#1781716267
exit
#1781706947
dir
#1781706950
cd compression
#1781706955
sbatch sbatch_benchmark.sh 
#1781706977
sq
#1781707122
sbatch sbatch_benchmark.sh 
#1781707139
sq
#1781713283
tpython 7.2_compression_pipeline_no_gather.py 
#1781713309
load_modules && start_gluon
#1781713314
load_module && start_gluon
#1781713316
tpython 7.2_compression_pipeline_no_gather.py 
#1781713854
tpython 7.2_compression_pipeline_no_gather.py \
#1781713858
tpython 7.2_compression_pipeline_no_gather.py 
#1781713996
tpython 9.1_find_max_shape.py 
#1781714188
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#1781714189
tpython 9.1_find_max_shape.py 
#1781715968
sbatch sbatch_benchmark.sh 
#1781716657
sbatch 9.2_sbatch_best_ratio.sh 
#1781716666
sbatch 9.3_sbatch_best_shape.sh 
#1781716751
sbatch 9.2_sbatch_best_ratio.sh 
#1781716753
sbatch 9.3_sbatch_best_shape.sh 
#1781717827
sq
#1781705168
debugjob
#1781811144
load_module && start_gluon
#1781811476
apptainer exec --nvccli ncu --set full -k "Profiling/best_shape/7.2" -o sparse_persistent_matmul_pipelined python 7.2_compression_pipeline_no_gather.py 
#1781811498
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -k "Profiling/best_shape/7.2" -o sparse_persistent_matmul_pipelined python 7.2_compression_pipeline_no_gather.py 
#1781811526
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -k "Profiling/best_shape/7.2" -o sparse_persistent_matmul_pipelined_kernel python 7.2_compression_pipeline_no_gather.py 
#1781811651
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -k sparse_persistent_matmul_pipelined_kernel -o Profiling/best_shape/7.2 python 7.2_compression_pipeline_no_gather.py 
#1781811686
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -k sparse_persistent_matmul_pipelined_kernel -o Profiling/best_shape/sparse python gluon_persistent.py 
#1781811715
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -k sparse_persistent_matmul_pipelined_kernel -o Profiling/best_shape/sparse python gluon_pipeline.py 
#1781811743
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -k persistent_matmul_pipelined_kernel -o Profiling/best_shape/dense python gluon_pipeline.py 
#1781812348
tpython 9.1_find_max_shape.py 
#1781812359
tpython 9.1_find_max_shape.py 7.2 16
#1781812555
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -k persistent_matmul_pipelined_kernel -o Profiling/max_shape/dense python gluon_pipeline.py 
#1781812566
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k persistent_matmul_pipelined_kernel -o Profiling/max_shape/dense python gluon_pipeline.py 
#1781812594
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/sparse python gluon_pipeline.py 
#1781812621
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/7.2 7.2_compression_pipeline_no_gather.py 
#1781812633
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/7.2 python 7.2_compression_pipeline_no_gather.py 
#1781814029
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/sparse python gluon_pipeline.py 
#1781814058
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/7.2 python 7.2_compression_pipeline_no_gather.py 
#1781814085
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k persistent_matmul_pipelined_kernel -o Profiling/max_shape/dense python gluon_pipeline.py 
#1781814202
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/7.2 python 7.2_compression_pipeline_no_gather.py 
#1781814496
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k persistent_matmul_pipelined_kernel -o Profiling/max_shape/dense python gluon_pipeline.py 
#1781814520
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/sparse python gluon_pipeline.py 
#1781818100
tpython 3D_single_tile_no_gather.py 
#1781818105
tpython 3D_single_tile_no_gather.py > ptx.txt
#1781818246
exit
#1781795765
load_module && start_gluon
#1781795772
tpython 3D_single_tile_no_gather.py 
#1781796088
tpython 8.3_benchmark_single_tile.py 
#1781796157
tpython 3d
#1781796161
tpython 3D_single_tile_no_gather.py 
#1781796224
tpython 3F_single_tile_reduce.py 
#1781796231
tpython 3G_single_tile_no_gather_nz_convert_layout.py 
#1781796348
tpython 8.3_benchmark_single_tile.py 
#1781798525
tpython 3D_single_tile_no_gather.py > ptx.txt
#1781798755
tpython 3D_single_tile_no_gather.py 
#1781798799
tpython 3D_single_tile_no_gather.py > ptx.txt
#1781800461
tpython 3D_single_tile_no_gather.py 
#1781800503
tpython 3D_single_tile_no_gather.py > ptx.txt
#1781801176
tpython 8.3_benchmark_single_tile.py 
#1781801214
tpython 3D_single_tile_no_gather.py > ptx.txt
#1781801371
debugjob
#1781803067
load_module && start_gluon
#1781803077
sbatch 7.2_benchmark_604042.out 
#1781803082
sbatch 7.2_s
#1781803083
sbatch 7.2_sbatch_benchmark.sh 
#1781803091
sbatch 9.2_sbatch_best_ratio.sh 
#1781803097
sbatch 9.3_sbatch_best_shape.sh 
#1781803113
sq
#1781810588
tpython 3D_single_tile_no_gather.py 
#1781810791
tpython 8.3_benchmark_single_tile.py 
#1781825561
sbatch 7.2_sbatch_benchmark.sh 
#1781825575
sq
#1781825579
sdq
#1781825581
sq
#1781825583
squeue
#1781825597
sq
#1781825621
sbatch 7.2_sbatch_benchmark.sh 
#1781825623
sq
#1781836657
pip install torch
#1781826267
cd /home/notming/links/scratch/compression
#1781826268
cat -n /home/notming/links/scratch/compression/3I_single_tile_ptx.py | head -90
#1781826797
load_module && start_gluon && tpython -c "import triton.experimental.gluon.language as gl; import inspect; print(inspect.getsource(gl.inline_asm_elementwise))"
#1781826818
cd /home/notming/links/scratch/compression
#1781826819
load_module && start_gluon && tpython -c "
#1781826819
import triton.experimental.gluon.language as gl
#1781826819
import inspect
#1781826819

#1781826819
# Check what to_tensor does in the gluon semantic module
#1781826819
from triton.experimental.gluon.language import _core as core
#1781826819
# Try to find the semantic module
#1781826819
import triton.language._core as tl_core
#1781826819
print('=== to_tensor ===')
#1781826819
print(inspect.getsource(tl_core._semantic.to_tensor))
#1781826819
" 2>&1 | tail -30
#1781826835
cd /home/notming/links/scratch/compression
#1781826837
load_module && start_gluon && tpython -c "
#1781826837
from triton.experimental.gluon.language._core import distributed_tensor
#1781826837
import inspect
#1781826837
print(inspect.getsource(distributed_tensor))
#1781826837
" 2>&1 | head -80
#1781826857
cd /home/notming/links/scratch/compression
#1781826858
load_module && start_gluon && tpython -c "
#1781826858
from triton.experimental.gluon.language import _core
#1781826858
import inspect
#1781826858
# Find the 'values' attribute usage
#1781826858
src = inspect.getsource(_core)
#1781826858
for i, line in enumerate(src.split('\n')):
#1781826858
    if 'values' in line.lower() and ('asm' in line.lower() or 'inline' in line.lower() or 'def ' in line.lower()):
#1781826858
        print(f'{i}: {line}')
#1781826858
" 2>&1 | tail -30
#1781826879
cd /home/notming/links/scratch/compression
#1781826880
load_module && start_gluon && tpython -c "
#1781826880
from triton.experimental.gluon.language import _core
#1781826880
import inspect
#1781826880
# Find where 'values' is accessed as an attribute
#1781826880
src = inspect.getsource(_core)
#1781826880
for i, line in enumerate(src.split('\n')):
#1781826880
    if '.values' in line:
#1781826880
        print(f'{i}: {line}')
#1781826880
" 2>&1 | tail -40
#1781826897
cd /home/notming/links/scratch/compression
#1781826898
load_module && start_gluon && tpython -c "
#1781826898
import triton.language.core as core
#1781826898
import inspect
#1781826898
src = inspect.getsource(core)
#1781826898
lines = src.split('\n')
#1781826898
for i, line in enumerate(lines):
#1781826898
    if 'to_tensor' in line and 'def ' in line:
#1781826898
        # print surrounding context
#1781826898
        for j in range(max(0,i-2), min(len(lines), i+20)):
#1781826898
            print(f'{j}: {lines[j]}')
#1781826898
        print('---')
#1781826898
" 2>&1 | tail -50
#1781826915
cd /home/notming/links/scratch/compression
#1781826916
load_module && start_gluon && tpython -c "
#1781826916
# Search for 'values' attribute access in the full traceback path
#1781826916
import triton.language.semantic as sem
#1781826916
import inspect
#1781826916
src = inspect.getsource(sem)
#1781826916
lines = src.split('\n')
#1781826916
for i, line in enumerate(lines):
#1781826916
    if '.values' in line:
#1781826916
        print(f'{i}: {line}')
#1781826916
" 2>&1 | tail -30
#1781826933
cd /home/notming/links/scratch/compression
#1781826934
load_module && start_gluon && tpython -c "
#1781826934
# The error says 'tensor' object has no attribute 'values'
#1781826934
# Let's trace where this comes from in the Gluon semantic override
#1781826934
from triton.experimental.gluon.language import _semantic
#1781826934
import inspect
#1781826934
src = inspect.getsource(_semantic)
#1781826934
lines = src.split('\n')
#1781826934
for i, line in enumerate(lines):
#1781826934
    if '.values' in line:
#1781826934
        print(f'{i}: {line}')
#1781826934
" 2>&1 | tail -30
#1781826950
cd /home/notming/links/scratch/compression
#1781826951
load_module && start_gluon && tpython -c "
#1781826951
from triton.experimental.gluon.language import _semantic
#1781826951
import inspect
#1781826951
src = inspect.getsource(_semantic)
#1781826951
lines = src.split('\n')
#1781826951
# Show the to_tensor method and binary_op_type_checking_impl
#1781826951
for i, line in enumerate(lines):
#1781826951
    if 'to_tensor' in line or 'binary_op_type_checking' in line:
#1781826951
        for j in range(max(0,i-1), min(len(lines), i+15)):
#1781826951
            print(f'{j}: {lines[j]}')
#1781826951
        print('---')
#1781826951
" 2>&1 | tail -60
#1781886439
load_module && start_gluon
#1781886446
tpython 3J_single_tile_ptx_match.py 
#1781886487
tpython 3J_single_tile_ptx_match.py  > ptx_correctness.txt 
#1781890555
tpython 3J_single_tile_ptx_match.py > ptx_correctness.txt 
#1781890729
tpython 3J_single_tile_ptx_match.py 
#1781890795
tpython 3J_single_tile_ptx_match.py > ptx_correctness.txt 
#1781892628
tpython ptx_check.py
#1781893059
tpython 3J_single_tile_ptx_match.py > ptx_correctness.txt 
#1781893077
tpython ptx_check.py
#1781893172
tpython 3J_single_tile_ptx_match.py > ptx_correctness.txt 
#1781893191
tpython ptx_check.py
#1781893253
tpython 3J_single_tile_ptx_match.py > ptx_correctness.txt  && tpython ptx_check.py
#1781893974
tpython 3J_single_tile_ptx_match.py > ptx_correctness.txt  && tpython ptx_check.py > ptx_check.py
#1781893980
tpython 3J_single_tile_ptx_match.py > ptx_correctness.txt  && tpython ptx_check.py > ptx_checktxt
#1781893986
tpython 3J_single_tile_ptx_match.py > ptx_correctness.txt  && tpython ptx_check.py > ptx_check.txt
#1781892601
cd /home/notming/links/scratch/compression
#1781892602
load_module && start_gluon && tpython ptx_check.py
#1781892843
cd /home/notming/links/scratch/compression
#1781892845
python simulate.py
#1781892901
python simulate.py
#1781893011
load_module && start_gluon && tpython 3J_single_tile_ptx_match.py > ptx_correctness.txt && tpython ptx_check.py
#1781893069
cd /home/notming/links/scratch/compression
#1781893071
load_module && start_gluon && tpython 3J_single_tile_ptx_match.py
#1781893157
cd /home/notming/links/scratch/compression
#1781893158
load_module && start_gluon && tpython test_shfl.py
#1781893315
cd /home/notming/links/scratch/compression
#1781893316
load_module && start_gluon && tpython test_trans.py
#1781893352
cd /home/notming/links/scratch/compression
#1781893354
load_module && start_gluon && tpython 3J_single_tile_ptx_match.py
#1781923368
load_module && start_gluon
#1781923468
tpython 3J_single_tile_ptx_match.py > ptx_correctness.txt && tpython ptx_check.py > ptx_check.txt
#1781929647
debugjob
#1781923619
cd /home/notming/links/scratch/compression
#1781923621
tpython 3J_single_tile_ptx_match.py
#1781923790
grep -A 5 "WGMMA:" ptx_correctness.txt
#1781923806
grep -n "T0:2" ptx_correctness.txt
#1781923872
grep -A 20 "Kernel Output Sample:" ptx_correctness.txt
#1781923883
grep -A 10 "PyTorch Reference Sample:" ptx_correctness.txt
#1781924010
cat << 'EOF' > test_t0.py
#1781924010
import re

#1781924010
import ast

#1781924010


#1781924010
def parse_tensor(lines):

#1781924010
    text = " ".join(lines)

#1781924010
    match = re.search(r'tensor\(\[\[(.*?)\]\]', text)

#1781924010
    if not match: return []

#1781924010
    rows_str = re.findall(r'\[(.*?)\]', text)

#1781924010
    tensor = []

#1781924010
    for row_str in rows_str:

#1781924010
        nums = re.findall(r'[-+]?\d*\.\d+(?:e[-+]?\d+)?', row_str)

#1781924010
        if nums:

#1781924010
            tensor.append([float(n) for n in nums])

#1781924010
    return tensor

#1781924010


#1781924010
with open('ptx_correctness.txt', 'r') as f:

#1781924010
    content = f.read()

#1781924010


#1781924010
# print whether T0:4 is mismatched in the actual strings

#1781924010
kernel_out = []

#1781924010
ref_out = []

#1781924010
current = None

#1781924010
for line in content.split('\n'):

#1781924010
    if line.startswith('Kernel Output'): current = 'k'

#1781924010
    elif line.startswith('PyTorch Ref'): current = 'r'

#1781924010
    elif line.startswith('WGMMA'): current = 'w'

#1781924010
    elif current == 'k' and '[' in line: kernel_out.append(line)

#1781924010
    elif current == 'r' and '[' in line: ref_out.append(line)

#1781924010


#1781924010
print("Kernel parsed:", len(parse_tensor(kernel_out)))

#1781924010
print("Ref parsed:", len(parse_tensor(ref_out)))

#1781924010


#1781924010
EOF

#1781924010
tpython test_t0.py
#1781924019
tpython test_t0.py
#1781924151
tpython 3J_single_tile_ptx_match.py > ptx_correctness.txt && tpython ptx_check.py > ptx_check.txt && cat ptx_check.txt
#1781924170
cd /home/notming
#1781924172
load_module && start_gluon -c "cd scratch/compression && tpython 3J_single_tile_ptx_match.py > ptx_correctness.txt && tpython ptx_check.py > ptx_check.txt && cat ptx_check.txt"
#1781926322
cd /home/notming/links/scratch/compression
#1781926323
cat << 'EOF' > test_trans.py
#1781926323
import triton

#1781926323
import triton.language as tl

#1781926323
import gluon.language as gl

#1781926323


#1781926323
@gl.jit

#1781926323
def test_kernel():

#1781926323
    a = gl.zeros((64, 16), dtype=gl.int32)

#1781926323
    # Try reshaping and transposing

#1781926323
    a_trans = a.reshape(4, 2, 8, 16).trans(0, 2, 3, 1)

#1781926323
    a0, a1 = a_trans.split()

#1781926323
    gl.static_print(a0.shape)

#1781926323


#1781926323
EOF

#1781926323
tpython test_trans.py
#1781926347
cat /home/notming/.gemini/antigravity-ide/brain/0b7068d1-d2db-47ce-8088-7bf07762b07e/.system_generated/tasks/task-90.log
#1781926370
cd /home/notming
#1781926372
cat << 'EOF' > test_trans.py
#1781926372
import triton.language as tl

#1781926372
import gluon.language as gl

#1781926372


#1781926372
@gl.jit

#1781926372
def test_kernel():

#1781926372
    y0 = gl.zeros((64, 4), dtype=gl.int32)

#1781926372
    # Try transposing so that Row 8 is the last dimension!

#1781926372
    # 64 = 4 * 2 * 8. So (4, 2, 8, 4). Transpose to (4, 8, 4, 2).

#1781926372
    y0_r = y0.reshape(4, 2, 8, 4).trans(0, 2, 3, 1)

#1781926372
    y0_r0, y0_r8 = y0_r.split()

#1781926372


#1781926372
EOF

#1781926372
tpython test_trans.py
#1781926448
cat << 'EOF' > test_ptx_8.py
#1781926448
import triton.language as tl

#1781926448
import gluon.language as gl

#1781926448


#1781926448
TRANSPOSE_8_PTX = """

#1781926448
    .reg .pred %p1, %p2;

#1781926448
    .reg .b32 %lane, %t1, %t2;

#1781926448
    mov.u32 %lane, %laneid;

#1781926448


#1781926448
    and.b32 %t1, %lane, 1;

#1781926448
    setp.ne.b32 %p1, %t1, 0;

#1781926448
    and.b32 %t2, %lane, 2;

#1781926448
    setp.ne.b32 %p2, %t2, 0;

#1781926448


#1781926448
    .reg .b32 %r0_0, %r1_0, %r2_0, %r3_0;

#1781926448
    .reg .b32 %r0_8, %r1_8, %r2_8, %r3_8;

#1781926448


#1781926448
    .reg .b32 %s1_0, %s3_0, %e1_0, %e3_0;

#1781926448
    selp.b32 %s1_0, $8, $9, %p1;

#1781926448
    selp.b32 %s3_0, $10, $11, %p1;

#1781926448
    shfl.sync.bfly.b32 %e1_0, %s1_0, 1, 0x1f, 0xffffffff;

#1781926448
    shfl.sync.bfly.b32 %e3_0, %s3_0, 1, 0x1f, 0xffffffff;

#1781926448


#1781926448
    @%p1  mov.b32 %r0_0, %e1_0;

#1781926448
    @!%p1 mov.b32 %r1_0, %e1_0;

#1781926448
    @%p1  mov.b32 %r2_0, %e3_0;

#1781926448
    @!%p1 mov.b32 %r3_0, %e3_0;

#1781926448
    @!%p1 mov.b32 %r0_0, $8;

#1781926448
    @%p1  mov.b32 %r1_0, $9;

#1781926448
    @!%p1 mov.b32 %r2_0, $10;

#1781926448
    @%p1  mov.b32 %r3_0, $11;

#1781926448


#1781926448
    .reg .b32 %s2_0, %s3_new_0, %e2_0, %e3_new_0;

#1781926448
    selp.b32 %s2_0, %r0_0, %r2_0, %p2;

#1781926448
    selp.b32 %s3_new_0, %r1_0, %r3_0, %p2;

#1781926448
    shfl.sync.bfly.b32 %e2_0, %s2_0, 2, 0x1f, 0xffffffff;

#1781926448
    shfl.sync.bfly.b32 %e3_new_0, %s3_new_0, 2, 0x1f, 0xffffffff;

#1781926448


#1781926448
    .reg .b32 %r0_final_0, %r1_final_0, %r2_final_0, %r3_final_0;

#1781926448
    @%p2  mov.b32 %r0_final_0, %e2_0;

#1781926448
    @!%p2 mov.b32 %r2_final_0, %e2_0;

#1781926448
    @%p2  mov.b32 %r1_final_0, %e3_new_0;

#1781926448
    @!%p2 mov.b32 %r3_final_0, %e3_new_0;

#1781926448
    @!%p2 mov.b32 %r0_final_0, %r0_0;

#1781926448
    @%p2  mov.b32 %r1_final_0, %r1_0;

#1781926448
    @!%p2 mov.b32 %r2_final_0, %r2_0;

#1781926448
    @%p2  mov.b32 %r3_final_0, %r3_0;

#1781926448


#1781926448
    .reg .b32 %s1_8, %s3_8, %e1_8, %e3_8;

#1781926448
    selp.b32 %s1_8, $12, $13, %p1;

#1781926448
    selp.b32 %s3_8, $14, $15, %p1;

#1781926448
    shfl.sync.bfly.b32 %e1_8, %s1_8, 1, 0x1f, 0xffffffff;

#1781926448
    shfl.sync.bfly.b32 %e3_8, %s3_8, 1, 0x1f, 0xffffffff;

#1781926448


#1781926448
    @%p1  mov.b32 %r0_8, %e1_8;

#1781926448
    @!%p1 mov.b32 %r1_8, %e1_8;

#1781926448
    @%p1  mov.b32 %r2_8, %e3_8;

#1781926448
    @!%p1 mov.b32 %r3_8, %e3_8;

#1781926448
    @!%p1 mov.b32 %r0_8, $12;

#1781926448
    @%p1  mov.b32 %r1_8, $13;

#1781926448
    @!%p1 mov.b32 %r2_8, $14;

#1781926448
    @%p1  mov.b32 %r3_8, $15;

#1781926448


#1781926448
    .reg .b32 %s2_8, %s3_new_8, %e2_8, %e3_new_8;

#1781926448
    selp.b32 %s2_8, %r0_8, %r2_8, %p2;

#1781926448
    selp.b32 %s3_new_8, %r1_8, %r3_8, %p2;

#1781926448
    shfl.sync.bfly.b32 %e2_8, %s2_8, 2, 0x1f, 0xffffffff;

#1781926448
    shfl.sync.bfly.b32 %e3_new_8, %s3_new_8, 2, 0x1f, 0xffffffff;

#1781926448


#1781926448
    .reg .b32 %r0_final_8, %r1_final_8, %r2_final_8, %r3_final_8;

#1781926448
    @%p2  mov.b32 %r0_final_8, %e2_8;

#1781926448
    @!%p2 mov.b32 %r2_final_8, %e2_8;

#1781926448
    @%p2  mov.b32 %r1_final_8, %e3_new_8;

#1781926448
    @!%p2 mov.b32 %r3_final_8, %e3_new_8;

#1781926448
    @!%p2 mov.b32 %r0_final_8, %r0_8;

#1781926448
    @%p2  mov.b32 %r1_final_8, %r1_8;

#1781926448
    @!%p2 mov.b32 %r2_final_8, %r2_8;

#1781926448
    @%p2  mov.b32 %r3_final_8, %r3_8;

#1781926448


#1781926448
    mov.b32 $0, %r0_final_0;

#1781926448
    mov.b32 $1, %r0_final_8;

#1781926448
    mov.b32 $2, %r1_final_0;

#1781926448
    mov.b32 $3, %r1_final_8;

#1781926448
    mov.b32 $4, %r2_final_0;

#1781926448
    mov.b32 $5, %r2_final_8;

#1781926448
    mov.b32 $6, %r3_final_0;

#1781926448
    mov.b32 $7, %r3_final_8;

#1781926448
"""

#1781926448


#1781926448
@gl.jit

#1781926448
def test_kernel():

#1781926448
    a = gl.zeros((64, 16), dtype=gl.int32)

#1781926448
    (y,) = gl.inline_asm_elementwise(

#1781926448
        TRANSPOSE_8_PTX,

#1781926448
        "=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r",

#1781926448
        [a],

#1781926448
        dtype=(gl.int32,),

#1781926448
        is_pure=True,

#1781926448
        pack=8,

#1781926448
    )

#1781926448
    gl.static_print(y.shape)

#1781926448
EOF

#1781926448
tpython test_ptx_8.py
#1781970576
load_module && start_gluon
#1781970583
sbatch 7.4_sbatch_benchmark.sh 
#1781970586
sq
#1781969745
debugjob
#1782057501
git init -b main
#1782057523
git add .
#1782057895
git rm --cached -r .venv
#1782057916
cd $HOMW
#1782057920
cd $HOME
#1782057921
dir
#1782057943
~bashrc.sh
#1782058005
cd $SCRATCH
#1782058048
dir
#1782058052
git add .
#1782058542
git submodule deinit -f gluon_spmm
#1782058547
dir
#1782058628
git add .
#1782058806
git init -b main
#1782058846
git add note
#1782058866
nano .git
#1782058884
cd .git
#1782058886
dir
#1782058899
nano index.lock 
#1782058906
cd ..
#1782058925
git add note
#1782058955
rm -f .git/index.lock 
#1782058957
git add note
#1782059018
rm -f .git/index.lock 
#1782059197
git config --global user.email "yinming.chan@gmail.com"
#1782059209
git config --global user.name " notming11"
#1782059242
git remote add origin https://github.com/notming11/GPU-compression-kernel.git
#1782059248
git push -u origin main
#1782704545
load_module && start_gluon
#1782704057
cd compression/
#1782704063
cd sbatch_sh/
#1782704066
cd trillium
#1782704072
sbatch 7_benchmark_all.sh 
#1782704102
sq
#1782704262
scancel 622526
#1782704265
sq
#1782704333
sbatch 7_benchmark_all.sh 
#1782704342
scancel 622529
#1782704349
sbatch 7_benchmark_all.sh 
#1782704352
sq
#1782794795
debugjob
#1782794981
debugjob --exclude=trig0001
#1782832733
load_module && start_gluon
#1782832738
tpython gluon_ws_dense.py 
#1782832925
tpython 7.6_compression_ws.py 
#1782832964
tpython gluon_ws_dense.py 
#1782832146
cd /home/notming/links/scratch/compression
#1782832148
cat << 'EOF' > /home/notming/links/scratch/compression/7.6_compression_ws.py
#1782832148
# This will be replaced with the full code.

#1782832148
EOF

#1782832492
cat << 'EOF' > /home/notming/links/scratch/compression/7.6_compression_ws.py
#1782832492
import torch

#1782832492
import triton

#1782832492


#1782832492
from triton.experimental import gluon

#1782832492
from triton.experimental.gluon import language as gl

#1782832492


#1782832492
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

#1782832492
from triton.language.core import _aggregate as aggregate

#1782832492


#1782832492
from triton.experimental.gluon.language.nvidia.hopper import (

#1782832492
    tma,

#1782832492
    mbarrier,

#1782832492
    fence_async_shared,

#1782832492
)

#1782832492


#1782832492
from common import (

#1782832492
    WGMMA,

#1782832492
    GroupedPersistentTileScheduler

#1782832492
)

#1782832492
import os

#1782832492


#1782832492
from prune import prune_2_4

#1782832492
from compress_2_4 import compress_dense_to_sparse

#1782832492


#1782832492
@aggregate

#1782832492
class SparsePartitionArgs:

#1782832492
    a_pruned_desc: tma.tensor_descriptor

#1782832492
    b_desc: tma.tensor_descriptor

#1782832492
    c_desc: tma.tensor_descriptor

#1782832492
    

#1782832492
    a_pruned_bufs: gl.shared_memory_descriptor

#1782832492
    a_comp_bufs: gl.shared_memory_descriptor

#1782832492
    e_bufs: gl.shared_memory_descriptor

#1782832492
    b_bufs: gl.shared_memory_descriptor

#1782832492
    

#1782832492
    a_pruned_empty_bars: gl.shared_memory_descriptor

#1782832492
    a_pruned_ready_bars: gl.shared_memory_descriptor

#1782832492
    

#1782832492
    a_comp_empty_bars: gl.shared_memory_descriptor

#1782832492
    a_comp_ready_bars: gl.shared_memory_descriptor

#1782832492


#1782832492
    b_empty_bars: gl.shared_memory_descriptor

#1782832492
    b_ready_bars: gl.shared_memory_descriptor

#1782832492
    

#1782832492
    acc_bufs: gl.shared_memory_descriptor

#1782832492
    acc_empty_bars: gl.shared_memory_descriptor

#1782832492
    acc_ready_bars: gl.shared_memory_descriptor

#1782832492
    

#1782832492
    SUBTILE_FACTOR: gl.constexpr

#1782832492
    num_warps_compute: gl.constexpr

#1782832492
    num_warps_compress: gl.constexpr

#1782832492


#1782832492
    @gluon.constexpr_function

#1782832492
    def __init__(self, a_pruned_desc, b_desc, c_desc,

#1782832492
                 a_pruned_bufs, a_comp_bufs, e_bufs, b_bufs,

#1782832492
                 a_pruned_empty_bars, a_pruned_ready_bars,

#1782832492
                 a_comp_empty_bars, a_comp_ready_bars,

#1782832492
                 b_empty_bars, b_ready_bars,

#1782832492
                 acc_bufs, acc_empty_bars, acc_ready_bars, 

#1782832492
                 SUBTILE_FACTOR, num_warps_compute, num_warps_compress):

#1782832492
        self.a_pruned_desc = a_pruned_desc

#1782832492
        self.b_desc = b_desc

#1782832492
        self.c_desc = c_desc

#1782832492
        self.a_pruned_bufs = a_pruned_bufs

#1782832492
        self.a_comp_bufs = a_comp_bufs

#1782832492
        self.e_bufs = e_bufs

#1782832492
        self.b_bufs = b_bufs

#1782832492
        self.a_pruned_empty_bars = a_pruned_empty_bars

#1782832492
        self.a_pruned_ready_bars = a_pruned_ready_bars

#1782832492
        self.a_comp_empty_bars = a_comp_empty_bars

#1782832492
        self.a_comp_ready_bars = a_comp_ready_bars

#1782832492
        self.b_empty_bars = b_empty_bars

#1782832492
        self.b_ready_bars = b_ready_bars

#1782832492
        self.acc_bufs = acc_bufs

#1782832492
        self.acc_empty_bars = acc_empty_bars

#1782832492
        self.acc_ready_bars = acc_ready_bars

#1782832492
        self.SUBTILE_FACTOR = gl.constexpr(SUBTILE_FACTOR)

#1782832492
        self.num_warps_compute = gl.constexpr(num_warps_compute)

#1782832492
        self.num_warps_compress = gl.constexpr(num_warps_compress)

#1782832492


#1782832492
@aggregate

#1782832492
class Counter:

#1782832492
    index: gl.tensor

#1782832492
    phase: gl.tensor

#1782832492
    num_barriers: gl.constexpr

#1782832492


#1782832492
    @gluon.constexpr_function

#1782832492
    def __init__(self, index, phase, num_barriers):

#1782832492
        self.index = index

#1782832492
        self.phase = phase

#1782832492
        self.num_barriers = gl.constexpr(num_barriers)

#1782832492


#1782832492
    @gluon.jit

#1782832492
    def create(phase, num_barriers: gl.constexpr):

#1782832492
        return Counter(gl.to_tensor(0), gl.to_tensor(phase), num_barriers)

#1782832492


#1782832492
    @gluon.must_use_result

#1782832492
    @gluon.jit

#1782832492
    def next(self, pred=True):

#1782832492
        incr = self.index + gl.where(pred, 1, 0)

#1782832492
        rollover = incr == self.num_barriers

#1782832492
        index = gl.where(rollover, 0, incr)

#1782832492
        phase = gl.where(rollover, self.phase ^ 1, self.phase)

#1782832492
        return Counter(index, phase, self.num_barriers)

#1782832492


#1782832492
@gluon.jit

#1782832492
def _split_n(x, SUBTILE_FACTOR: gl.constexpr):

#1782832492
    split_count: gl.constexpr = SUBTILE_FACTOR.bit_length() - 1  # log2

#1782832492
    xs = (x, )

#1782832492
    for _ in gl.static_range(split_count):

#1782832492
        next_xs = ()

#1782832492
        for j in gl.static_range(len(xs)):

#1782832492
            x = xs[j]

#1782832492
            next_xs += x.reshape(x.shape[0], 2, x.shape[1] // 2).permute(0, 2, 1).split()

#1782832492
        xs = next_xs

#1782832492
    return xs

#1782832492


#1782832492
@gluon.jit

#1782832492
def sparse_matmul_load_partition(p, SchedulerImpl: gl.constexpr):

#1782832492
    BLOCK_M: gl.constexpr = p.a_pruned_desc.block_type.shape[0]

#1782832492
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]

#1782832492
    BLOCK_K: gl.constexpr = p.b_desc.block_type.shape[0]

#1782832492
    K = p.b_desc.shape[0]

#1782832492


#1782832492
    state_a = Counter.create(1, p.a_pruned_empty_bars.shape[0])

#1782832492
    state_b = Counter.create(1, p.b_empty_bars.shape[0])

#1782832492
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)

#1782832492


#1782832492
    for idx in range(scheduler.get_num_tiles()):

#1782832492
        pid_m, pid_n = scheduler.get_tile(idx)

#1782832492
        off_m = pid_m * BLOCK_M

#1782832492
        off_n = pid_n * BLOCK_N

#1782832492


#1782832492
        for k in range(0, K, BLOCK_K):

#1782832492
            bar_a = p.a_pruned_ready_bars.index(state_a.index)

#1782832492
            bar_b = p.b_ready_bars.index(state_b.index)

#1782832492
            mbarrier.wait(p.a_pruned_empty_bars.index(state_a.index), state_a.phase)

#1782832492
            mbarrier.wait(p.b_empty_bars.index(state_b.index), state_b.phase)

#1782832492


#1782832492
            mbarrier.expect(bar_a, p.a_pruned_desc.block_type.nbytes)

#1782832492
            mbarrier.expect(bar_b, p.b_desc.block_type.nbytes)

#1782832492
            

#1782832492
            tma.async_copy_global_to_shared(p.a_pruned_desc, [off_m, k], bar_a, p.a_pruned_bufs.index(state_a.index))

#1782832492
            tma.async_copy_global_to_shared(p.b_desc, [k, off_n], bar_b, p.b_bufs.index(state_b.index))

#1782832492
            

#1782832492
            state_a = state_a.next()

#1782832492
            state_b = state_b.next()

#1782832492


#1782832492
@gluon.jit

#1782832492
def sparse_matmul_compress_partition(p, SchedulerImpl: gl.constexpr):

#1782832492
    BLOCK_M: gl.constexpr = p.a_pruned_desc.block_type.shape[0]

#1782832492
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]

#1782832492
    BLOCK_K: gl.constexpr = p.b_desc.block_type.shape[0]

#1782832492
    K = p.b_desc.shape[0]

#1782832492


#1782832492
    num_warps = p.num_warps_compress

#1782832492


#1782832492
    state_a = Counter.create(0, p.a_pruned_empty_bars.shape[0])

#1782832492
    state_comp = Counter.create(1, p.a_comp_empty_bars.shape[0])

#1782832492
    

#1782832492
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)

#1782832492


#1782832492
    a_warp_bases: gl.constexpr = [[16, 0], [32, 0]] if num_warps == 4 else ([[16, 0], [32, 0], [0, 0]] if num_warps == 8 else [[16, 0], [32, 0], [0, 0], [0, 0]])

#1782832492
    a_shape: gl.constexpr = [64, 64]

#1782832492
    a_pruned_reg_layout: gl.constexpr = gl.DistributedLinearLayout(

#1782832492
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]], 

#1782832492
        lane_bases=[[0, 16], [0, 32], [1, 0], [2, 0], [4, 0]], 

#1782832493
        warp_bases=a_warp_bases, 

#1782832493
        block_bases=[], 

#1782832493
        shape=a_shape

#1782832493
    )

#1782832493


#1782832493
    for _ in range(scheduler.get_num_tiles()):

#1782832493
        for _ in range(0, K, BLOCK_K):

#1782832493
            mbarrier.wait(p.a_pruned_ready_bars.index(state_a.index), state_a.phase)

#1782832493
            mbarrier.wait(p.a_comp_empty_bars.index(state_comp.index), state_comp.phase)

#1782832493


#1782832493
            a_pruned_smem = p.a_pruned_bufs.index(state_a.index)

#1782832493
            a_pruned = a_pruned_smem.load(a_pruned_reg_layout)

#1782832493


#1782832493
            a_grouped = a_pruned.reshape(BLOCK_M, BLOCK_K // 4, 2, 2)

#1782832493
            a_even, a_odd = a_grouped.split()

#1782832493


#1782832493
            a0, a2 = a_even.split()

#1782832493
            a1, a3 = a_odd.split()

#1782832493


#1782832493
            idx0 = (~(a0 != 0) & (a1 != 0)) | ((~(a0 != 0) & ~(a1 != 0)) << 1)

#1782832493
            idx1 = (((a0 != 0) & (a1 != 0)) | (~(a0 != 0) & ~(a1 != 0)) | (a3 != 0)) | (((~(a0 != 0) & (a1 != 0)) | ~(a1 != 0)) << 1)

#1782832493


#1782832493
            nz0 = gl.where(idx0 == 0, a0, gl.where(idx0 == 1, a1, gl.where(idx0 == 2, a2, a3)))

#1782832493
            nz1 = gl.where(idx1 == 0, a0, gl.where(idx1 == 1, a1, gl.where(idx1 == 2, a2, a3)))

#1782832493


#1782832493
            a_compressed = gl.join(nz0, nz1)

#1782832493
            a_compressed = a_compressed.reshape(BLOCK_M, BLOCK_K // 2)

#1782832493


#1782832493
            meta_4 = idx0 | (idx1 << 2)

#1782832493


#1782832493
            meta_grouped = meta_4.reshape(BLOCK_M, BLOCK_K // 16, 2, 2)

#1782832493
            meta_even, meta_odd = meta_grouped.split()

#1782832493


#1782832493
            mn0, mn2 = meta_even.split()

#1782832493
            mn1, mn3 = meta_odd.split()

#1782832493


#1782832493
            mn0 = mn0.to(gl.int16)

#1782832493
            mn1 = mn1.to(gl.int16)

#1782832493
            mn2 = mn2.to(gl.int16)

#1782832493
            mn3 = mn3.to(gl.int16)

#1782832493


#1782832493
            meta = mn0 | (mn1 << 4) | (mn2 << 8) | (mn3 << 12)

#1782832493
            meta_reshaped = meta.reshape(BLOCK_M // 16, 2, 8, BLOCK_K // 64, 4)

#1782832493
            meta_reordered = meta_reshaped.permute(0, 3, 2, 4, 1).reshape(BLOCK_M // 16, BLOCK_K)

#1782832493


#1782832493
            a_comp_smem = p.a_comp_bufs.index(state_comp.index)

#1782832493
            e_smem = p.e_bufs.index(state_comp.index)

#1782832493


#1782832493
            a_comp_smem.store(a_compressed)

#1782832493
            e_smem.store(meta_reordered)

#1782832493


#1782832493
            fence_async_shared()

#1782832493


#1782832493
            mbarrier.arrive(p.a_pruned_empty_bars.index(state_a.index), count=1)

#1782832493
            mbarrier.arrive(p.a_comp_ready_bars.index(state_comp.index), count=1)

#1782832493


#1782832493
            state_a = state_a.next()

#1782832493
            state_comp = state_comp.next()

#1782832493


#1782832493
@gluon.jit

#1782832493
def store_acc_to_smem_subtile(p, mma, acc_state):

#1782832493
    mma = mma.wait_num_outstanding(0)

#1782832493
    acc, mma = mma.take_result()

#1782832493
    accs = _split_n(acc, p.SUBTILE_FACTOR)

#1782832493


#1782832493
    for i in gl.static_range(p.SUBTILE_FACTOR):

#1782832493
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)

#1782832493
        c_buf = p.acc_bufs.index(acc_state.index)

#1782832493


#1782832493
        c_buf.store(accs[i].to(p.c_desc.dtype))

#1782832493
        fence_async_shared()

#1782832493
        mbarrier.arrive(p.acc_ready_bars.index(acc_state.index), count=1)

#1782832493
        acc_state = acc_state.next()

#1782832493


#1782832493
    return acc_state

#1782832493


#1782832493
@gluon.jit

#1782832493
def sparse_matmul_compute_partition(p, SchedulerImpl: gl.constexpr):

#1782832493
    BLOCK_M: gl.constexpr = p.a_pruned_desc.block_type.shape[0]

#1782832493
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]

#1782832493
    BLOCK_K: gl.constexpr = p.b_desc.block_type.shape[0]

#1782832493
    K = p.b_desc.shape[0]

#1782832493
    dtype: gl.constexpr = p.a_pruned_desc.dtype

#1782832493


#1782832493
    state_comp = Counter.create(0, p.a_comp_empty_bars.shape[0])

#1782832493
    state_b = Counter.create(0, p.b_empty_bars.shape[0])

#1782832493
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])

#1782832493


#1782832493
    release_comp = Counter.create(0, p.a_comp_empty_bars.shape[0])

#1782832493
    release_b = Counter.create(0, p.b_empty_bars.shape[0])

#1782832493


#1782832493
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)

#1782832493


#1782832493
    outstanding_mmas: gl.constexpr = 0

#1782832493
    global_k_iter = 0

#1782832493


#1782832493
    for _ in range(scheduler.get_num_tiles()):

#1782832493
        mma = WGMMA.initialize(dtype, BLOCK_M, BLOCK_N, p.num_warps_compute, sparse=True)

#1782832493


#1782832493
        for _ in range(0, K, BLOCK_K):

#1782832493
            mbarrier.wait(p.a_comp_ready_bars.index(state_comp.index), state_comp.phase)

#1782832493
            mbarrier.wait(p.b_ready_bars.index(state_b.index), state_b.phase)

#1782832493


#1782832493
            mma = mma.wait_num_outstanding(outstanding_mmas)

#1782832493
            mma = mma.issue_async_sparse_mma(p.a_comp_bufs.index(state_comp.index), p.e_bufs.index(state_comp.index), p.b_bufs.index(state_b.index))

#1782832493


#1782832493
            if global_k_iter >= outstanding_mmas + 1:

#1782832493
                mbarrier.arrive(p.a_comp_empty_bars.index(release_comp.index), count=1)

#1782832493
                mbarrier.arrive(p.b_empty_bars.index(release_b.index), count=1)

#1782832493
                release_comp = release_comp.next()

#1782832493
                release_b = release_b.next()

#1782832493


#1782832493
            state_comp = state_comp.next()

#1782832493
            state_b = state_b.next()

#1782832493
            global_k_iter += 1

#1782832493


#1782832493
        acc_state = store_acc_to_smem_subtile(p, mma, acc_state)

#1782832493


#1782832493
@gluon.jit

#1782832493
def sparse_matmul_store_partition(p, SchedulerImpl: gl.constexpr):

#1782832493
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]

#1782832493
    SPLIT_N: gl.constexpr = p.c_desc.block_type.shape[1]

#1782832493
    BLOCK_N: gl.constexpr = SPLIT_N * p.SUBTILE_FACTOR

#1782832493


#1782832493
    state = Counter.create(0, p.acc_empty_bars.shape[0])

#1782832493
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)

#1782832493


#1782832493
    num_buffers: gl.constexpr = 2

#1782832493
    outstanding_stores: gl.constexpr = 1

#1782832493
    store_iter = 0

#1782832493


#1782832493
    for idx in range(scheduler.get_num_tiles()):

#1782832493
        pid_m, pid_n = scheduler.get_tile(idx)

#1782832493
        off_m, off_n = pid_m * BLOCK_M, pid_n * BLOCK_N

#1782832493


#1782832493
        for i in gl.static_range(p.SUBTILE_FACTOR):

#1782832493
            mbarrier.wait(p.acc_ready_bars.index(state.index), state.phase)

#1782832493
            c_buf = p.acc_bufs.index(state.index)

#1782832493


#1782832493
            tma.async_copy_shared_to_global(p.c_desc, [off_m, off_n + i * SPLIT_N], c_buf)

#1782832493


#1782832493
            if store_iter >= outstanding_stores:

#1782832493
                tma.store_wait(outstanding_stores)

#1782832493
                empty_idx = (store_iter - outstanding_stores) % num_buffers

#1782832493
                mbarrier.arrive(p.acc_empty_bars.index(empty_idx), count=1)

#1782832493


#1782832493
            state = state.next()

#1782832493
            store_iter += 1

#1782832493


#1782832493
    tma.store_wait(0)

#1782832493


#1782832493
@gluon.jit

#1782832493
def sparse_matmul_warp_specialized_kernel(

#1782832493
    a_pruned_desc, b_desc, c_desc, SchedulerImpl: gl.constexpr,

#1782832493
    M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,

#1782832493
    num_buffers: gl.constexpr, SUBTILE_FACTOR: gl.constexpr,

#1782832493
    num_warps_compute: gl.constexpr, num_warps_compress: gl.constexpr):

#1782832493
    

#1782832493
    dtype: gl.constexpr = a_pruned_desc.dtype

#1782832493


#1782832493
    a_pruned_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_pruned_desc.block_type.shape, a_pruned_desc.layout)

#1782832493
    

#1782832493
    a_comp_shape = [a_pruned_desc.block_type.shape[0], a_pruned_desc.block_type.shape[1] // 2]

#1782832493
    a_comp_layout = gl.NVMMASharedLayout.get_default_for(a_comp_shape, dtype)

#1782832493
    a_comp_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_comp_shape, a_comp_layout)

#1782832493
    

#1782832493
    e_shape = [a_pruned_desc.block_type.shape[0] // 16, a_pruned_desc.block_type.shape[1]]

#1782832493
    e_layout = gl.NVMMASharedLayout.get_default_for(e_shape, gl.int16)

#1782832493
    e_bufs = gl.allocate_shared_memory(gl.int16, [num_buffers] + e_shape, e_layout)

#1782832493
    

#1782832493
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)

#1782832493


#1782832493
    a_pruned_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())

#1782832493
    a_pruned_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())

#1782832493
    

#1782832493
    a_comp_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())

#1782832493
    a_comp_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())

#1782832493
    

#1782832493
    b_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())

#1782832493
    b_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())

#1782832493
    

#1782832493
    for i in gl.static_range(num_buffers):

#1782832493
        mbarrier.init(a_pruned_empty_bars.index(i), count=1)

#1782832493
        mbarrier.init(a_pruned_ready_bars.index(i), count=1)

#1782832493
        mbarrier.init(a_comp_empty_bars.index(i), count=1)

#1782832493
        mbarrier.init(a_comp_ready_bars.index(i), count=1)

#1782832493
        mbarrier.init(b_empty_bars.index(i), count=1)

#1782832493
        mbarrier.init(b_ready_bars.index(i), count=1)

#1782832493


#1782832493
    acc_bufs = gl.allocate_shared_memory(dtype, [2] + c_desc.block_type.shape, c_desc.layout)

#1782832493
    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())

#1782832493
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())

#1782832493


#1782832493
    for i in gl.static_range(2):

#1782832493
        mbarrier.init(acc_empty_bars.index(i), count=1)

#1782832493
        mbarrier.init(acc_ready_bars.index(i), count=1)

#1782832493


#1782832493
    p = SparsePartitionArgs(a_pruned_desc, b_desc, c_desc,

#1782832493
                            a_pruned_bufs, a_comp_bufs, e_bufs, b_bufs,

#1782832493
                            a_pruned_empty_bars, a_pruned_ready_bars,

#1782832493
                            a_comp_empty_bars, a_comp_ready_bars,

#1782832493
                            b_empty_bars, b_ready_bars,

#1782832493
                            acc_bufs, acc_empty_bars, acc_ready_bars,

#1782832493
                            SUBTILE_FACTOR, num_warps_compute, num_warps_compress)

#1782832493


#1782832493
    gl.warp_specialize([

#1782832493
        (sparse_matmul_compute_partition, (p, SchedulerImpl)),

#1782832493
        (sparse_matmul_compress_partition, (p, SchedulerImpl)),

#1782832493
        (sparse_matmul_load_partition, (p, SchedulerImpl)),

#1782832493
        (sparse_matmul_store_partition, (p, SchedulerImpl)),

#1782832493
    ], [num_warps_compress, 1, 1], [64, 64, 24, 24])

#1782832493


#1782832493
def sparse_matmul_get_configs(pre_hook=None):

#1782832493
    def valid(BM, BN, BK, warps_compute, warps_compress, buffers, SF):

#1782832493
        if (BN // SF) < 16: return False

#1782832493
        return True

#1782832493
    

#1782832493
    return [

#1782832493
        triton.Config(

#1782832493
            {

#1782832493
                "BLOCK_SIZE_M": BM,

#1782832493
                "BLOCK_SIZE_N": BN,

#1782832493
                "BLOCK_SIZE_K": BK,

#1782832493
                "num_buffers": buffers,

#1782832493
                "SUBTILE_FACTOR": SF,

#1782832493
                "num_warps_compress": warps_compress,

#1782832494
                "num_warps_compute": warps_compute,

#1782832494
            },

#1782832494
            num_warps=warps_compute + warps_compress + 1 + 1,

#1782832494
            pre_hook=pre_hook,

#1782832494
        )

#1782832494
        for BM in (128, 256)

#1782832494
        for BN in (128, 256)

#1782832494
        for BK in (64, 128)

#1782832494
        for warps_compute in (4, 8)

#1782832494
        for warps_compress in (4, 8)

#1782832494
        for buffers in (3, 4)

#1782832494
        for SF in (1, 2)

#1782832494
        if valid(BM, BN, BK, warps_compute, warps_compress, buffers, SF)

#1782832494
    ]

#1782832494


#1782832494
def sparse_matmul_tma_set_block_size_hook(nargs):

#1782832494
    block_m = nargs["BLOCK_SIZE_M"]

#1782832494
    block_n = nargs["BLOCK_SIZE_N"]

#1782832494
    block_k = nargs["BLOCK_SIZE_K"]

#1782832494
    split_n = nargs["BLOCK_SIZE_N"] // nargs["SUBTILE_FACTOR"]

#1782832494


#1782832494
    nargs["a_pruned_desc"].block_shape = [block_m, block_k]

#1782832494
    nargs["b_desc"].block_shape = [block_k, block_n]

#1782832494
    nargs["c_desc"].block_shape = [block_m, split_n]

#1782832494


#1782832494
    nargs["a_pruned_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["a_pruned_desc"].block_shape, gl.float16)

#1782832494
    nargs["b_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["b_desc"].block_shape, gl.float16)

#1782832494
    nargs["c_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["c_desc"].block_shape, gl.float16)

#1782832494


#1782832494
sparse_ws_kernel = triton.autotune(

#1782832494
    configs=sparse_matmul_get_configs(pre_hook=sparse_matmul_tma_set_block_size_hook),

#1782832494
    key=["M", "N", "K"],

#1782832494
)(sparse_matmul_warp_specialized_kernel)

#1782832494


#1782832494
def run_sparse_ws_matmul(A_pruned, B):

#1782832494
    M, K = A_pruned.shape[0], A_pruned.shape[1]

#1782832494
    N = B.shape[1]

#1782832494


#1782832494
    c = torch.empty((M, N), device=A_pruned.device, dtype=torch.float16)

#1782832494
    dummy_block = [1, 1]

#1782832494
    dummy_layout_f16 = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)

#1782832494
    

#1782832494
    a_pruned_desc = TensorDescriptor.from_tensor(A_pruned, dummy_block, dummy_layout_f16)

#1782832494
    b_desc = TensorDescriptor.from_tensor(B, dummy_block, dummy_layout_f16)

#1782832494
    c_desc = TensorDescriptor.from_tensor(c, dummy_block, dummy_layout_f16)

#1782832494


#1782832494
    def grid(meta):

#1782832494
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

#1782832494
        num_pid = triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"])

#1782832494
        return (min(num_sms, num_pid), )

#1782832494
    

#1782832494
    sparse_ws_kernel[grid](a_pruned_desc, b_desc, c_desc, GroupedPersistentTileScheduler(8), M, N, K)

#1782832494


#1782832494
    return c

#1782832494


#1782832494
if __name__ == "__main__":

#1782832494
    os.environ["MLIR_ENABLE_DUMP"]="1"

#1782832494
    os.environ["MLIR_DUMP_PATH"] = "./MLIR_DUMP/7.6"

#1782832494
    os.environ["TRITON_ALWAYS_COMPILE"]="1"

#1782832494


#1782832494
    M, N, K = 16384, 4096, 49152

#1782832494


#1782832494
    print(f"Testing 7.6_compression_ws: M={M}, N={N}, K={K}...", end=" ", flush=True)

#1782832494


#1782832494
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)

#1782832494
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)

#1782832494


#1782832494
    A_pruned = prune_2_4(A)

#1782832494


#1782832494
    C = run_sparse_ws_matmul(A_pruned, B)

#1782832494
    C_ref = A_pruned @ B

#1782832494


#1782832494
    torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-1)

#1782832494
    print("PASSED")

#1782832494


#1782832494
EOF

#1782832544
load_module && start_gluon && tpython 7.6_compression_ws.py
#1782832622
cd /home/notming/links/scratch/compression
#1782832623
load_module && start_gluon && tpython 7.6_compression_ws.py
#1782832698
cd /home/notming/links/scratch/compression
#1782832700
load_module && start_gluon && tpython 7.6_compression_ws.py
#1782832756
cd /home/notming/links/scratch/compression
#1782832758
load_module && start_gluon && tpython 7.6_compression_ws.py
#1782832811
cd /home/notming/links/scratch/compression
#1782832812
load_module && start_gluon && tpython 7.6_compression_ws.py
#1782832904
cd /home/notming/links/scratch/compression
#1782832905
load_module && start_gluon && tpython 7.6_compression_ws.py
#1782833002
cd /home/notming/links/scratch/compression
#1782833003
load_module && start_gluon && tpython 7.6_compression_ws.py
#1782833058
cd /home/notming/links/scratch/compression
#1782833060
load_module && start_gluon && tpython 7.6_compression_ws.py
#1782833101
cd /home/notming/links/scratch/compression
#1782833103
load_module && start_gluon && tpython 7.6_compression_ws.py
#1783030956
cd compression/
#1783030964
sbatch sbatch_sh/trillium/7.6_benchmark_ws.sh 
#1783030969
sq
#1783031685
sbatch sbatch_sh/trillium/7.6_benchmark_ws.sh 
#1783031841
sq
#1783031850
scancel 629376
#1783031852
sq
#1783042327
sbatch sbatch_sh/trillium/7.6_benchmark_ws.sh 
#1783042331
sq
#1783020522
debugjob
#1783024444
cd /home/notming/links/scratch/compression
#1783024445
tpython 7.6_compression_ws.py
#1783099619
cd compression/
#1783099624
sbatch sbatch_sh/
#1783099631
sbatch sbatch_sh/trillium/7.6_benchmark_ws.sh 
#1783099650
scancel 631572
#1783099652
sq
#1783101440
scancel 631571
#1783101448
sbatch sbatch_sh/trillium/7.6_benchmark_ws.sh 
#1783101451
sq
#1783096224
debugjobD
#1783137225
cd compression/
#1783137235
sbatch sbatch_sh/trillium/Profile.sh 
#1783263962
sq
#1783263996
scancel 635001
#1783264001
scancel 635002
#1783264099
sq
#1783273219
git reset --soft HEAD~1
#1783273566
git config --global http.postBuffer 1572864000
#1783273572
git push origin main
#1783263919
cd compression/
#1783263951
sbatch sbatch_sh/trillium/Profile.sh 
#1783386410
git reset --soft HEAD~1
#1783386463
git pull
#1783386473
git fetch
#1783386474
git pull
#1783386510
git reset --soft HEAD~1
#1783386691
debugjob
#1783388271
git reset --soft HEAD~1
#1783388372
git lfs install
#1783388380
git lfs track "*ncu-rep"
#1783391290
sq
#1783392240
git reset --soft HEAD~1
#1783392244
git lfs track "*ncu-rep"
#1783392251
git lfs install
#1783392254
git lfs track "*ncu-rep"
#1783392294
sq
#1783392298
git reset --soft HEAD~1
#1783443470
debugjob
#1783561619
load_module && start_gluon
#1783561625
tpython 3F_single_tile_reduce.py 
#1783563568
sq
#1783571336
git --reset HEAD
#1783571347
git reset --soft HEAD~1
#1783571485
debugjob
#1783538493
load_module && start_gluon
#1783538502
tpython 3K_single_tile_4_registers.py 
#1783538509
tpython 3K_single_tile_4_registers.py > layout.txt
#1783556322
tpython 3F_single_tile_reduce.py 
#1783556471
debugjob
#1783564259
sbatch sbatch_sh/trillium/7.4_sbatch_benchmark.sh 
#1783564408
sq
#1783571111
tpython 3F_single_tile_reduce.py 
#1783572311
load_module && start_gluon
#1783572395
apptainer exec --nvccli $SCRATCH.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/7.3 python 7.3_compression_pipeline_reduce.py 
#1783572407
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/7.3 python 7.3_compression_pipeline_reduce.py 
#1783572447
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/7.5 python 7.5_compression_pipeline_no_ldmatrix.py 
#1783572489
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/sparse python gluon_pipeline.py 
#1783572533
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k persistent_matmul_pipelined_kernel -o Profiling/max_shape/dense python gluon_pipeline.py 
#1783618746
load_module && start_gluon
#1783618753
tpython 7.3_compression_pipeline_reduce.py 
#1783618760
tpython 7.3_compression_pipeline_reduce.py > layout.txt
#1783619510
tpython 3F_single_tile_reduce.py 
#1783620776
tpython 7.3_compression_pipeline_reduce.py 
#1783624087
tpython 3F_single_tile_reduce.py 
#1783624131
tpython 7.3_compression_pipeline_reduce.py > layout.txt
#1783624341
sbatch sbatch_sh/trillium/7.4_sbatch_benchmark.sh 
#1783624344
sq
#1783624442
debugjob
#1783647898
tpython 7.3_compression_pipeline_reduce.py 
#1783648010
tpython 7.3_compression_pipeline_reduce.py > per_thread.txt
#1783648412
tpython 7.3_compression_pipeline_reduce.py
#1783648493
tpython 8.7_benchmark_persistent.py 7.3 16
#1783650956
debugjob
#1783619192
cd /home/notming/links/scratch/compression
#1783619193
load_module && start_gluon && tpython test_script.py
#1783619226
cd /home/notming/links/scratch/compression
#1783619227
load_module && start_gluon && tpython test_script.py
#1783619277
cd /home/notming/links/scratch/compression
#1783619278
load_module && start_gluon && tpython 7.3_compression_pipeline_reduce.py
#1783623508
cd /home/notming/links/scratch/compression
#1783623509
load_module && start_gluon && tpython test_script2.py
#1783623555
cd /home/notming/links/scratch/compression
#1783623556
load_module && start_gluon && tpython test_script2.py
#1783623601
cd /home/notming/links/scratch/compression
#1783623602
load_module && start_gluon && tpython test_script3.py
#1783623702
cd /home/notming/links/scratch/compression
#1783623703
load_module && start_gluon && tpython 3F_single_tile_reduce.py
#1783623871
cd /home/notming/links/scratch/compression
#1783623872
load_module && start_gluon && tpython test_script4.py
#1783623969
cd /home/notming/links/scratch/compression
#1783623971
grep -n "meta_4 =" 7.3_compression_pipeline_reduce.py
#1783625611
load_module && start_gluon
#1783625620
sbatch sbatch_sh/trillium/7.4_sbatch_benchmark.sh 
#1783625913
sq
#1783825537
load_module && start_gluon
#1783825551
tpython 7.3_compression_pipeline_reduce.py 
#1783825982
debugjob --exclude=trig0001
#1783833556
load_module && start_gluon
#1783833576
sbatch sbatch_sh/trillium/7.4_sbatch_benchmark.sh 
#1783833589
scancel 658718
#1783833595
sbatch sbatch_sh/trillium/7.4_sbatch_benchmark.sh 
#1783833597
sq
#1783871957
load_module && start_gluon
#1783871999
sbatch sbatch_sh/trillium/7.4_sbatch_benchmark.sh 
#1783874034
sq
#1783913696
scancel 662452
#1783913794
sbatch sbatch_sh/trillium/7.4_sbatch_benchmark.sh 
#1783914968
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/7.3.1 python 7.3.1_compression_pipeline_reduce_interlayout.py 
#1783915095
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/7.5 python 7.5_compression_pipeline_no_ldmatrix.py 
#1783915446
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/7.3.1 python 7.3.1_compression_pipeline_reduce_interlayout.py 
#1783915476
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/7.5 python 7.5_compression_pipeline_no_ldmatrix.py 
#1783915520
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/sparse python gluon_pipeline.py 
#1783915581
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k persistent_matmul_pipelined_kernel -o Profiling/max_shape/dense python gluon_pipeline.py 
#1783915857
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/7.3 python 7.3_compression_pipeline_reduce.py 
#1783916962
sq
#1783917711
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_persistent_matmul_pipelined_kernel -o Profiling/max_shape/7.5.1 python 7.5.1_compression_pipeline_ldmatrix.py 
#1783917785
sbatch sbatch_sh/trillium/7.4_sbatch_benchmark.sh 
#1783918640
sq
#1783869656
load_module && start_gluon
#1783869670
sbatch sbatch_sh/trillium/7.3
#1783869673
sbatch sbatch_sh/trillium/7.4_sbatch_benchmark.sh 
#1783870149
scancel 660333
#1783870230
debugjob
#1783878494
tpython 7.3_temp.py 
#1783878619
debugjob
#1783909999
tpython test_script.py 
#1783911382
tpython 3F.1_single_tile_reduce_interlayout.py 
#1783911492
tpython 7.3.1_compression_pipeline_reduce_interlayout.py 
#1783911670
sbatch sbatch_sh/trillium/7.4_sbatch_benchmark.sh 
#1783912782
tpython 3H_single_tile_no_gather_better_layout.py 
#1783913064
tpython 7.5_compression_pipeline_no_ldmatrix.py 
#1783913147
debugjob
#1783955961
sq
#1783955964
squeue
#1784954184
scancel --me
#1784954205
load_module && start_gluon
#1784954227
sbatch sbatch_sh/trillium/7.6.2_benchmark.sh 
#1784954293
sbatch sbatch_sh/trillium/7.6.1_benchmark.sh 
#1784924750
load_module && start_gluon
#1784924769
sbatch sbatch_sh/trillium/7.6.2_benchmark.sh 
#1784924779
sq
#1784924785
sq --start
#1784924831
sq
#1784924968
sbatch sbatch_sh/trillium/7.6.2_benchmark.sh 
#1784924970
sq
#1784925173
sq -i 60
#1785080254
sq
#1785081619
scancel --me
#1785081671
load_module && start_gluon
#1785081739
sbatch sbatch_sh/trillium/7.7.1_benchmark.sh 
#1785081747
sq
#1785117787
git reset --soft HEAD~1
#1785127329
sq
#1785131745
sbatch sbatch_sh/trillium/7.6.2_benchmark.sh 
#1785074873
debugjob
#1785125185
sq -i 60
#1785125188
debugjob
#1785168421
load_module && start_gluon
#1785168436
tpython 7.6.2_compression_ws_register_buffer.py --tune
#1785168453
tpython 7.6.2_compression_ws_register_buffer.py 8192
#1785168467
tpython 8.7.1_benchmark_ws.py 7.6.2 8192
#1785170452
load_module && start_gluon
#1785170470
sbatch sbatch_sh/trillium/7.6.2_benchmark.sh 
#1785170550
sq
#1785172428
scancel 680226
#1785172431
cancel 680227
#1785172435
scancel 680227
#1785173921
sbatch sbatch_sh/trillium/7.6.2_benchmark.sh 
#1785173946
sq
#1785175432
scancel --me
#1785179426
sq
#1785167342
sq
#1785167353
scancle --me
#1785167367
scancel --me
#1785168395
debugjob
#1785180683
cd /home/notming/links/scratch/compression
#1785180684
tpython 7.6.3_compress_ws_load_and_compress.py
#1785180691
cd /scratch/notming/compression
#1785180692
tpython 7.6.3_compress_ws_load_and_compress.py
#1785180726
tpython 7.6.3_compress_ws_load_and_compress.py
#1785195251
debugjob
#1785249715
load_module && start_gluon
#1785249748
tpython 7.6.3_compress_ws_load_and_compress.py 
#1785251401
tpython 7.6.2_compression_ws_register_buffer.py 
#1785251474
exit
#1785257786
clear
#1785257974
load_module && start_gluon
#1785257987
debugjob
#1785258709

#1785259159
debugjob
#1785259172
tpython 7.7.1_ws_seperate_warp_4_buf.py 
#1785259312
tpython 7.7.1_ws_seperate_warp_4_buf.py --tune
#1785259437
nvidia-smi
#1785259624
apptainer exec $SCRATCH/sparse.sif python -i 7.7.1_ws_seperate_warp_4_buf.py 
#1785259666
apptainer exec $SCRATCH/sparse.sif python -i 7.7.1_ws_seperate_warp_4_buf.py > ptx_dump.txt
#1785259759
apptainer exec $SCRATCH/sparse.sif python -i 7.7.1_ws_seperate_warp_4_buf.py --tune > ptx_dump.txt
#1785259887
apptainer exec --nvccli $SCRATCH/sparse.sif python -i 7.7.1_ws_seperate_warp_4_buf.py --tune > ptx_dump.txt
#1785259945
tpython -i 7.7.1_ws_seperate_warp_4_buf.py --tune
#1785259965
tpython 7.7.1_ws_seperate_warp_4_buf.py --tune
#1785260008
nvidia-smi
#1785260062
tpython 7.7.1_ws_seperate_warp_4_buf.py --tune
#1785260148
tpython -i 7.7.1_ws_seperate_warp_4_buf.py --tune
#1785260951
tpython 7.7.1_ws_seperate_warp_4_buf.py 
#1785248546
sq
#1785248643
squeue -O priority
#1785248656
squeue -O priority,user,jobid
#1785248667
squeue -O sq
#1785248668
sq
#1785248706
squeue --partition=debug -O jobid,user,priority
#1785248733
squeue --sort=p --partition=debug -O jobid,user,priority
#1785248760
squeue --sort=p --partition=debug -O jobid,user,priority,status
#1785248787
squeue --sort=p --partition=debug -O jobid,username,priority,state
#1785248810
sq --start
#1785250073
sq
#1785258213
sq --start
#1785258217
squeue --sort=p --partition=debug -O jobid,username,priority,state
#1785275433
load_module && start_gluon
#1785275613
tpython 7.6.3_compress_ws_2_partition.py 
#1785275658
tpython 8.7.1_benchmark_ws.py 7.6.3 8192
#1785276171
tpython 7.7.1_ws_seperate_warp_4_buf.py --tune
#1785276939
tpython 8.7.1_benchmark_ws.py 7.7.1 8192
#1785277826
tpython 8.7.1_benchmark_ws.py 7.7.1 8192 > ptx_dump.txt
#1785278361
tpython 7.7.1_ws_seperate_warp_4_buf.py --tune
#1785278522
tpython 8.7.1_benchmark_ws.py 7.7.1 8192
#1785278951
tpython 8.7.1_benchmark_ws.py 7.6.2 16384
#1785279145
tpytyhon 7.6.2_compression_ws_register_buffer.py 
#1785279154
tpython 7.6.2_compression_ws_register_buffer.py 
#1785279363
tpython 7.7.1_ws_seperate_warp_4_buf.py 
#1785279606
tpython 7.7.1_ws_seperate_warp_4_buf.py --tune
#1785280221
tpython 7.7.1_ws_seperate_warp_4_buf.py 
#1785280471
tpython 7.7.1_ws_seperate_warp_4_buf.py --tune
#1785280642
tpython 7.7.1_ws_seperate_warp_4_buf.py 
#1785281636
tpython 8.7.1_benchmark_ws.py 7.7.1 8192
#1785275406
debugjob
#1785282383
load_module && start_gluon
#1785282401
rm -f compiler_scratch
#1785282409
rmdir -f compiler_scratch
#1785282416
rmdir --help
#1785282432
rmdir --force compiler_scratch
#1785282437
rmdir  compiler_scratch
#1785282446
cd compiler_scratch/
#1785282448
cd .triton_cache/
#1785282451
cd MIH4X24CEAJDWYGXCZ4EKW6DTWSUG2ZBSBPV7NGI62HAUKJ7F7BQ/
#1785282452
dir
#1785282466
rm -f .nfs622b6833f4e105da00010b72 
#1785282483
cd ../../..
#1785282486
debugjob
#1785291870
tpython 7.7.1_ws_seperate_warp_4_buf.py 
#1785291947
tpython 7.7.1_ws_seperate_warp_4_buf.py > ptx_dump.txt
#1785276080
load_module && start_gluon
#1785276095
sbatch sbatch_sh/trillium/7.6.2_benchmark.sh 
#1785276099
sq
#1785278386
scancel 685680
#1785279313
sbatch sbatch_sh/trillium/7.6.2_benchmark.sh 
#1785280302
sq
#1785282476
sbatch sbatch_sh/trillium/7.6.2_benchmark.sh 
#1785282952
sq
#1785337940
debugjob
#1785383341
load_module && start_gluon
#1785383390
sbatch sbatch_sh/trillium/7.6.4_benchmark.sh 
#1785337740
sbatch sbatch_sh/trillium/7.6.2_benchmark.sh 
#1785337754
load_module && start_gluon
#1785337756
sbatch sbatch_sh/trillium/7.6.2_benchmark.sh 
#1785337759
sq
#1785337991
sq --start
#1785338002
sq --start -i 15
#1785340432
sq
#1785346243
tpython 7.6.4_compression_ws_optimization.py --tune
#1785347168
tpython 7.6.4_compression_ws_optimization.py 
#1785349697
sbatch sbatch_sh/trillium/7.6.4_benchmark.sh 
#1785349712
sq
#1785349724
sq --start -i 60
#1785349790
sq
#1785352759
sq -i 60
#1785425400
load_module && start_gluon
#1785425422
tpython 8.7.1_benchmark_ws.py 7.6.3 8192
#1785425921
tpython 7.6.3_compress_ws_2_partition.py 
#1785425967
nvidia-smi
#1785425974
kill -9 2953865
#1785425990
tpython 7.6.3_compress_ws_2_partition.py 
#1785426226
tpython 8.7.1_benchmark_ws.py 7.6.3 8192
#1785426544
tpython 7.6.3_compress_ws_2_partition.py 
#1785426731
tpython 8.7.1_benchmark_ws.py 7.6.3 8192
#1785426753
tpython 7.6.3_compress_ws_2_partition.py 
#1785426857
tpython 8.7.1_benchmark_ws.py 7.6.3 8192
#1785426986
tpython 7.6.3_compress_ws_2_partition.py 
#1785427104
tpython 8.7.1_benchmark_ws.py 7.6.3 8192
#1785427391
tpython 7.8.2_prune_ws_2_partition.py 
#1785427439
tpython 8.7.1_benchmark_ws.py 7.8.2 8192
#1785427654
tpython 8.7.1_benchmark_ws.py 7.6.3 8192
#1785427975
tpython 8.7.1_benchmark_ws.py 7.8.2 8192
#1785428762
tpython 8.7.1_benchmark_ws.py 7.6.3 8192
#1785432148
tpython 7.6.3_compress_ws_2_partition.py 
#1785432432
quit
#1785432434
exit
#1785416930
debugjob
#1785425382
debugjob\
#1785425386
debugjob
#1785445015
load_module && start_gluon
#1785445075
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_matmul_warp_specialized_kernel -o Profiling/ws/7.6.3 python 7.6.3_compress_ws_2_partition.py 
#1785445136
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_matmul_warp_specialized_kernel -o Profiling/ws/7.8.2 python 7.8.2_prune_ws_2_partition.py 
#1785446549
debugjob
#1785462835
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_matmul_warp_specialized_kernel -o Profile/ws/7.6.4_ptx_nz python 7.6.4_compression_ws_optimization.py 
#1785462883
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_matmul_warp_specialized_kernel -o Profile/ws/7.6.4 python 7.6.4_compression_ws_optimization.py 
#1785462924
debugjob
#1785415727
sq
#1785434601
start_gluon
#1785434615
sbatch sbatch_sh/trillium/7.8.1_benchmark.sh 
#1785434715
sbatch sbatch_sh/trillium/7.6.3_benchmark.sh 
#1785434744
sbatch sbatch_sh/trillium/7.8.2_benchmark.sh 
#1785434747
sq
#1785454519
sbatch sbatch_sh/trillium/7.8.1_benchmark.sh 
#1785454522
sq
#1785416831
cd /home/notming/links/scratch/compression
#1785416832
load_module && start_gluon && tpython -c "
#1785416832
import torch
#1785416832
from prune import prune_2_4
#1785416832

#1785416832
M, K = 1024, 1024
#1785416832
A = torch.randn(M, K, device='cuda', dtype=torch.float16)
#1785416832

#1785416832
A_ref = prune_2_4(A)
#1785416832

#1785416832
A_g = A.view(M, K // 4, 4)
#1785416832
a0 = A_g[:, :, 0]
#1785416832
a1 = A_g[:, :, 1]
#1785416832
a2 = A_g[:, :, 2]
#1785416832
a3 = A_g[:, :, 3]
#1785416832

#1785416832
c01 = a0 > a1
#1785416832
c02 = a0 > a2
#1785416832
c03 = a0 > a3
#1785416832
c12 = a1 > a2
#1785416832
c13 = a1 > a3
#1785416832
c23 = a2 > a3
#1785416832

#1785416832
c10 = ~c01
#1785416832
c20 = ~c02
#1785416832
c30 = ~c03
#1785416832
c21 = ~c12
#1785416832
c31 = ~c13
#1785416832
c32 = ~c23
#1785416832

#1785416832
p0 = (c01 & c02) | (c01 & c03) | (c02 & c03)
#1785416832
p1 = (c10 & c12) | (c10 & c13) | (c12 & c13)
#1785416832
p2 = (c20 & c21) | (c20 & c23) | (c21 & c23)
#1785416832
p3 = (c30 & c31) | (c30 & c32) | (c31 & c32)
#1785416832

#1785416832
a0_p = torch.where(p0, a0, torch.zeros_like(a0))
#1785416832
a1_p = torch.where(p1, a1, torch.zeros_like(a1))
#1785416832
a2_p = torch.where(p2, a2, torch.zeros_like(a2))
#1785416832
a3_p = torch.where(p3, a3, torch.zeros_like(a3))
#1785416832

#1785416832
A_our = torch.stack([a0_p, a1_p, a2_p, a3_p], dim=-1).view(M, K)
#1785416832

#1785416832
assert torch.equal(A_ref, A_our), 'Mismatch found!'
#1785416832
print('SUCCESS: 100% Match between prune_2_4 and our logic!')
#1785416832
"
#1785416867
cd /home/notming/links/scratch/compression
#1785416868
load_module && start_gluon && tpython 7.8.1_pruned_ws.py --bm 128 --bn 256 --bk 64 --warps 8 --buffers 3 --sf 4
#1785427363
cd /home/notming/links/scratch/compression
#1785427365
load_module && start_gluon && tpython 7.8.2_prune_ws_2_partition.py --bm 128 --bn 256 --bk 64 --warps 8 --buffers 3 --sf 4
#1785526793
load_module && start_gluon
#1785526867
tpython 8.7.1_benchmark_ws.py 7.6.4 8192
#1785527761
tpython 7.6.4_compression_ws_optimization.py 
#1785528385
nvidia-smi
#1785528398
kill -9 2826722 2826980
#1785528563
tpython 7.6.4_compression_ws_optimization.py 
#1785528593
nvidia-smi
#1785528599
kill -9 2827140
#1785528722
tpython 7.6.4_compression_ws_optimization.py 
#1785528834
tpython 8.7.1_benchmark_ws.py 7.6.4 8192
#1785532825
tpython gluon_ws_dense
#1785532841
tpython gluon_ws_dense.py 
#1785533009
tpython gluon_ws_dense.py --tune
#1785533155
tpython 7.6.4_compression_ws_optimization.py 
#1785533309
tpython 7.6.4_compression_ws_optimization.py --tune
#1785533901
load_module && start_gluon
#1785533903
tpython 7.6.4_compression_ws_optimization.py --tune
#1785536253
tpython 8.7.1_benchmark_ws.py 7.6.4 8192
#1785533873
sq
#1785539789
start_gluon
#1785539800
sbatch sbatch_sh/trillium/7.6.4_benchmark.sh 
#1785539817
sq
#1785539898
sq --start
#1785539903
sq
#1785539915
sq --start
#1785539917
sq
#1785549804
scancel 698637
#1785525523
start_gluon
#1785525575
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_matmul_warp_specialized_kernel -o Profiling/ws/7.6.4_4_stage python 7.6.4_compression_ws_optimization.py 
#1785525893
load_module 
#1785525902
tpython 7.6.4_compression_ws_optimization.py 
#1785526753
debugjob
#1785644317
load_module && start_gluon
#1785644352
tpython 10.1_compress_acc.py 
#1785644867
tpython 7.8.1_prune_ws.py 
#1785645017
tpython 10.1_prune_acc.py 
#1785646005
tpython 10.1_prune_acc.py > test.txt
#1785717341
load_module && start_gluon
#1785717572
tpython 8.7.1_benchmark_ws.py 10.1 8192
#1785718864
tpython 10.1_prune_acc.py --tune
#1785718902
tpython 8.7.1_benchmark_ws.py 10.1 8192
#1785719119
tpython 10.1_prune_acc.py 
#1785719536
tpython 8.7.1_benchmark_ws.py 10.1 8192
#1785720057
tpython 10.1_prune_acc.py --tune
#1785720841
tpython 8.7.1_benchmark_ws.py 10.1 8192
#1785725798
load_module && start_gluon
#1785725804
tpython 8.7.1_benchmark_ws.py 10.1 49152
#1785701636
debugjob
#1785717258
tpython 10.1_prune_acc.py
#1785717272
load_module && start_gluon
#1785717277
tpython 10.1_prune_acc.py
#1785717327
debugjob
#1785725180
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "sparse_matmul_warp_specialized_kernel" -o Profiling/ws/10.1 python 10.1_prune_acc.py 
#1785725674
debugjob
#1785725742
debugjob --hrlp
#1785725748
debugjob --help
#1785725766
debugjob --account=rrg-mmehride
#1785718993
load_module && start_gluon
#1785719003
tpython 10.1_prune_acc.py 
#1785725774
sq
#1785725780
sq --start
#1785727232
sbatch sbatch_sh/trillium/10.1_benchmark_all.sh 
#1785727238
sq
#1785768740
load_module && start_gluon
#1785768745
sbatch sbatch_sh/trillium/10.1_benchmark_all.sh 
#1785768925
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_matmul_warp_specialized_kernel -o Profiling/49152/10.1 python 10.1_prune_acc.py 
#1785769026
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k sparse_matmul_warp_specialized_kernel -o Profiling/49152/sparse python gluon_ws_sparse.py 
#1785769133
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k matmul_warp_specialized_kernel -o Profiling/49152/dense python gluon_ws_dense.py 
#1785769631
tpython 10.1_prune_acc.py 
#1785769809
tpython gluon_ws_sparse.py 
#1785769831
tpython 10.1_prune_acc.py 
#1785794527
tpython 7.6.4_compression_ws_optimization.py 
#1785807432
tpython gluon_ws_sparse.py --tune
#1785817730
sq
#1785801103
/home/notming/links/scratch/.venv/bin/python
#1785802405
load_module && start_gluon
#1785802412
tpython 11.1_2_kernel_baseline.py --tune
#1785811250
debugjob
#1785818530
tpython 8.7.1_benchmark_ws.py 11.1 8192
#1785856818
load_module && start_gluon
#1785856827
tpython 8.7.1_benchmark_ws.py 11.1 8192
#1785891962
load_module && start_gluon
#1785891982
tpython 8.7.1_benchmark_ws.py 7.8.1 8192
#1785893011
tpython 8.7.1_benchmark_ws.py 11.1 8192
#1785893508
tpython 8.7.1_benchmark_ws.py 7.8.1 8192
#1785904882
load_module && start_gluon
#1785904891
tpython 8.11_benchmark_2_kernel.py 11.1 8192
#1785942207
load_modul && start_gluon
#1785942210
load_module && start_gluon
#1785942223
tpython 8.11_benchmark_2_kernel.py 11.1 128
#1785882757
load_module && start_gluon
#1785882764
tpython 8.7.1_benchmark_ws.py 11.1 8192
#1785883027
nvidia-smi
#1785883043
tpython 8.7.1_benchmark_ws.py 7.6.4 8192
#1785885836
tpython 8.7.1_benchmark_ws.py 11.1 8192
#1785886277
tpython 8.7.1_benchmark_ws.py 7.8.1 8192
#1785887517
nvidia-smi
#1785891651
tpython 8.7.1_benchmark_ws.py 7.8.1 8192
#1785891947
debugjob
#1785904295
pip install torchao
#1785904330
pip install mslk --index-url https://download.pytorch.org/whl/cu130
#1785904347
pip install mslk
#1785904356
pip update mslk
#1785904364
pip install update mslk
#1785904375
pip install apache-tvm-ffi
#1785904384
pip install nvidia-cutlass-dsl==4.5.2 nvidia-cutlass-dsl-libs-base==4.5.2 nvidia-cutlass-dsl-libs-cu13==4.5.2
#1785904400
pip install nvidia-cutlass-dsl==4.5.2 nvidia-cutlass-dsl-libs-base==4.5.2 nvidia-cutlass-dsl-libs-cu13
#1785904408
pip install nvidia-cutlass-dsl==4.5.2 nvidia-cutlass-dsl-libs-base==4.5.2
#1785904870
debugjob
#1785943149
squeu -u=hungshou
#1785943155
squeue -u=hungshou
#1785943174
squeu --help
#1785943179
squeue --help
#1785943201
squeue -A=rrg-mmehride
#1785943211
squeue -A=def-mmehride
#1785943419
squeue
#1785906161
load_module && start_gluon
#1785906181
sbatch sbatch_sh/trillium/11.1_benchmark.sh 
#1785906185
sq
#1785906194
sq --start
#1785906213
sq
#1785906239
scancel --me
#1785906259
sbatch sbatch_sh/trillium/11.1_benchmark.sh 
#1785906261
sq --start
#1785906266
sq
#1785964848
load_module && start_gluon
#1785964854
tpython 8.11_benchmark_2_kernel.py 11.1 8192
#1785988242
load_module && start_gluon
#1785988251
tpython 8.11.1_benchmark_pruning.py 11.1
#1785956882
debugjob
#1785984087
load_module && start_gluon
#1785984100
tpython 8.11_benchmark_2_kernel.py 11.1 8192
#1785986744
tpython 8.11.1_benchmark_pruning.py 11.1
#1785988234
debugjob
#1785988672
sbatch sbatch_sh/trillium/11.1_benchmark_pruning.sh 
#1785988675
sq
#1785988683
sq --start
#1785988692
sq
#1785961106
load_module && start_gluon
#1785961111
pip install torchao
#1785964215
pip install mslk --index-url https://download.pytorch.org/whl/cu130
#1785964222
pip install --pre mslk --index-url https://download.pytorch.org/whl/nightly/cu130
#1785964230
pip install apache-tvm-ffi
#1785964230
pip install nvidia-cutlass-dsl==4.5.2 nvidia-cutlass-dsl-libs-base==4.5.2 nvidia-cutlass-dsl-libs-cu13==4.5.2
#1785964272
pip install nvidia-cutlass-dsl-libs-cu13
#1785965533
sbatch sbatch_sh/trillium/11.1_benchmark.sh 
#1785965540
sq
#1786022038
load_module && start_gluon
#1786022046
tpython 8.11_benchmark_2_kernel.py 11.1 8192
#1786031017
sq
#1786022028
debugjob
#1786022418
load_module && start_gluon
#1786022463
sbatch sbatch_sh/trillium/11.1_benchmark.sh
#1786022465
sq
#1786022481
sq --start
#1786022492
sq
#1786070018
/home/notming/links/scratch/.venv/bin/python
#1786067942
load_module && start_gluon
#1786067960
tpython 11.1_2_kernel_baseline.py 
#1786067987
tpython 11.1_2_kernel_baseline.py --tune
#1786068216
tpython 8.11.1_benchmark_pruning.py 11.1
#1786069189
debugjob
#1786072977
load_module && start_gluon
#1786072987
sbatch sbatch_sh/trillium/11.1_benchmark_pruning.sh 
#1786073066
sq
#1786073071
sq --start
#1786073081
sq
#1786073142
sq --start
#1786073312
sq
#1786073385
sbatch sbatch_sh/trillium/11.1_benchmark_pruning.sh 
#1786073387
sq
#1786073397
sq -i 60
#1786074547
sq
#1786067913
cd /home/notming/links/scratch/compression
#1786067914
load_module && start_gluon && tpython 11.1_2_kernel_baseline.py
#1786067939
cd /home/notming/links/scratch/compression
#1786067940
load_module && start_gluon && tpython 11.1_2_kernel_baseline.py --tune
#1786070196
cd /home/notming/links/scratch/compression
#1786070198
load_module && start_gluon && tpython 8.11.1_benchmark_pruning.py 11.1
#1786377040
load_module && start_gluon
#1786377045
cd ../attention/
#1786377050
tpython gluon_attention_forward.py 
#1786377126
tpython gluon_attention_forward.py --tune
#1786398684
load_module && start_gluon
#1786398690
cd ../attention/
#1786398701
tpython gluon_attention_forward.py > test.txt
#1786376189
load_module && start_gluon
#1786376192
cd .. 
#1786376194
cd attention/
#1786376200
tpython gluon_attention_forward.py 
#1786376602
cd ..
#1786376605
cd compression
#1786376773
cd ..
#1786376776
cd attention
#1786376799
ln -s ../compression/common.py ./common.py
#1786376814
tpython gluon_attention_forward.py 
#1786376949
ln ../compression/common.py ./common.py
#1786376958
tpython gluon_attention_forward.py 
#1786377028
debugjob
#1786384927
cd ../compression/
#1786384942
tpython gluon_ws_dense.py 
#1786385093
tpython gluon_ws_dense.py > layout.txt
#1786385154
tpython gluon_ws_dense.py 
#1786385682
cd ../attention/
#1786385685
tpython gluon_attention_forward.py 
#1786398672
debugjob
#1786463446
load_module && start_gluon
#1786463449
cd ../s
#1786463453
cd ../attention/
#1786463461
tpython gluon_attention_forward.py > test.txt 
#1786469411
load_module && start_gluon
#1786469414
cd ../attention/
#1786469417
tpython gluon_attention_forward.py 
#1786469445
tpython gluon_attention_forward.py > test.txt 
#1786470383
load_module && start_gluon
#1786470389
cd ../attention/
#1786470393
tpython gluon_attention_forward.py > test.txt 
#1786470429
nvidia-smi
#1786470436
kill -9 2039492
#1786470446
tpython gluon_attention_forward.py 
#1786470470
nvidia-smi
#1786470474
kill -9 2041102
#1786507455
/home/notming/links/scratch/.venv/bin/python
#1786502734
load_module && start_gluon
#1786502737
cd ../attention/
#1786502745
tpython gluon_attention_forward.py > test.txt
#1786504252
nvidia-smi
#1786504258
kill -9 2766675
#1786504261
tpython gluon_attention_forward.py > test.txt
#1786504273
tpython gluon_attention_forward.py
#1786504544
tpython benchmark.py 
#1786504650
tpython gluon_attention_forward.py --tune
#1786507510
tpython gluon_attention_forward.py --tune > ptx_dump.txt
#1786508804
tpython benchmark.py 
#1786489365
sq
#1786491457
load_module && start_gluon
#1786491474
sbatch sbatch_sh/trillium/11.1_benchmark.sh 
#1786491480
sq
#1786499647
pip install torchao
#1786500609
tpython 8.11_benchmark_2_kernel.py 11.1
#1786500747
tpython 8.11_benchmark_2_kernel.py 11.1 8192
#1786501092
tpython 8.11.1_benchmark_pruning.py 11.1
#1786501567
tpython 8.11.2_benchmark_pruning_cuSPARSElt.py 11.1 
#1786502696
debugjob
#1786517430
cd ../attention/
#1786517433
tpython gluon_attention_forward.py 
#1786461631
load_module && start_gluon
#1786461641
cd ../attention/
#1786461647
tpython gluon_attention_forward.py 
#1786463435
debugjob
#1786476039
cd ../compression/
#1786476050
sbatch sbatch_sh/trillium/11.1_benchmark.sh 
#1786476054
sq
#1786486021
module spider cusparselt
#1786486079
module load cusparselt/0.9.11
#1786486094
module load cusparselt/0.9.1.1
#1786486220
nvcc main.cu -o hopper_compress     -arch=sm_90     -I/path/to/cusparselt/include     -L/path/to/cusparselt/lib64     -lcusparseLt     -lcudart
#1786486457
./hopper_compress 
#1786486683
export CUSPARSELT_INCLUDE=/path/to/cusparselt/include
#1786486683
export CUSPARSELT_LIB=/path/to/cusparselt/lib64
#1786486690
debugjob
#1786501584
sq
#1786544200
load_module && start_gluon
#1786544205
cd ../attention/
#1786544211
tpython gluon_fa3_forward.py 
#1786544284
nvidia-smi
#1786544291
kill -9 1092872
#1786544294
tpython gluon_attention_forward.py 
#1786544568
tpython gluon_fa3_forward.py 
#1786544954
nvidia-smi
#1786544960
kill -9 1096486
#1786545132
tpython gluon_fa3_forward.py 
#1786545651
tpython gluon_fa3_forward.py --tune
#1786545991
kill -9
#1786545996
kill -l
#1786546004
kill --all
#1786546009
nvidia-smi
#1786546015
kill -9 1126871
#1786546024
tpython gluon_fa3_forward.py --tune > ptx_dump.txt 
#1786546093
nvidia-smi
#1786546098
kill -i 1137622
#1786546103
kill -9 1137622
#1786546116
tpython gluon_fa3_forward.py --tune > ptx_dump.txt 
#1786546343
nvidia-smi
#1786546348
kill -o 1139906
#1786546352
kill -9 1139906
#1786547246
tpython gluon_fa3_forward.py --tune > ptx_dump.txt 
#1786550737
sq
#1786550760
load_module && start_gluon
#1786550766
cd ../attention/
#1786550772
tpython benchmark.py 
#1786551703
nvidia-smi
#1786551710
kill -9 3028559
#1786551750
tpython benchmark.py 
#1786552072
tpython gluon_fa3_forward.py 
#1786552138
tpython gluon_fa3_forward.py --tune
#1786552495
tpython benchmark.py 
#1786552897
nvidia-smi
#1786552905
kill -9 3039544
#1786552945
tpython benchmark.py 
#1786552995
nvidia-smi
#1786553003
kill -9 3040355
#1786553170
tpython benchmark.py 
#1786554908
nvidia-smi
#1786554917
kill -9 3041011 3058899
#1786554924
tpython gluon_fa3_forward.py 
#1786554954
nvidia-smi
#1786554963
kill -9 3060518
#1786555160
tpython gluon_fa3_forward.py 
#1786555365
nvidia-smi
#1786555372
kill -9 3063532
#1786555374
tpython gluon_fa3_forward.py 
#1786555531
tpython benchmark.py 
#1786555579
TORCH_USE_CUDA_DSA=1 tpython benchmark.py 
#1786555616
TORCH_USE_CUDA_DSA tpython benchmark.py 
#1786555634
tpython TORCH_USE_CUDA_DSA=1 benchmark.py 
#1786555649
TORCH_USE_CUDA_DSA=1 tpython benchmark.py 
#1786561373
cuda-gdb
#1786561157
load_module && start_gluon
#1786561160
cd ../attention/
#1786561167
tpython benchmark.py 
#1786561209
nvidia-smi
#1786561215
kill -9 1505025
#1786561224
tpython gluon_fa3_forward.py --tune
#1786561301
tpython benchmark.py 
#1786561466
kill
#1786561478
nvidia-smi
#1786561484
kill 1507697
#1786561486
nvidia-smi
#1786561490
kill -9 1507697
#1786561491
nvidia-smi
#1786561942
tpython benchmark.py 
#1786562137
nvidia-smi
#1786562143
kill -9 1514391
#1786562189
tpython gluon_fa3_forward.py --tune
#1786562308
tpython benchmark.py 
#1786544167
debugjob
#1786599679
sq
#1786636452
load_module & start_gluon
#1786636458
cd ../attention/
#1786636478
tpython gluon_attention_forward.py 
#1786636492
tpython gluon_attention_forward.py --tune
#1786636923
tpython benchmark.py --head-dim 64
#1786637261
tpython benchmark.py --head-dim 64 > benchmark_log_64.txt 
#1786637865
tpython benchmark.py --head-dim 128
#1786642900
load_module && start_gluon
#1786642911
cd ../attention/
#1786642922
tpython gluon_attention_forward.py 
#1786637411
start_gluon
#1786637415
cd ../attention/
#1786637427
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786637436
sq
#1786637446
scancel 760743
#1786637468
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786637473
sq
#1786643744
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786643757
sq
#1786644034
scancel 761040
#1786644110
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786644199
sq
#1786636435
debugjob
#1786641427
load_module && start_gluon
#1786641441
cd ../attention/
#1786641505
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o "Profiling/4_partition_4096_128" python gluon_fa3_forward.py 
#1786641638
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o "Profiling/3_partition_4096_128" python gluon_attention_forward.py 
#1786642025
apptainer exec --nvccli $SCRATCH/sparse.sif     ncu --set full -f     --nvtx --nvtx-include "PyTorch_SDPA_4096_128"     -o "Profiling/pytorch_sdpa_4096_128"     python profile_sdpa.py
#1786642046
apptainer exec --nvccli $SCRATCH/sparse.sif     ncu --set full -f     --nvtx --nvtx-include "PyTorch_SDPA_4096_128"     -o "Profiling/pytorch_sdpa_4096_128"     python pytorch_sdpa.py 
#1786642120
apptainer exec --nvccli $SCRATCH/sparse.sif     ncu --set full -f     --nvtx --nvtx-include "PyTorch_SDPA_4096_128/"     -o "Profiling/pytorch_sdpa_4096_128"     python pytorch_sdpa.py 
#1786642881
debugjob
#1786644304
tpython gluon_fa3_forward.py 
#1786662453
load_module && start_gluon
#1786662456
cd ../attention/
#1786662469
tpython gluon_fa3_forward.py --tune
#1786663899
tpython benchmark.py
#1786664397
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o 
#1786664418
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o "Profiling/4_partition_4096_128" python gluon_fa3_forward.py 
#1786664777
tpython benchmark.py --head-dim=64
#1786647914
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o "Profiling/4_partition_4096_128" python gluon_fa3_forward.py 
#1786647923
load_module && start_gluon
#1786647927
cd ../attention/
#1786647931
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o "Profiling/4_partition_4096_128" python gluon_fa3_forward.py 
#1786647970
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o "Profiling/3_partition_4096_128" python gluon_attention_forward.py 
#1786648019
apptainer exec --nvccli $SCRATCH/sparse.sif     ncu --set full -f     --nvtx --nvtx-include "PyTorch_SDPA_4096_128/"     -o "Profiling/pytorch_sdpa_4096_128"     python pytorch_sdpa.py 
#1786649043
tpython gluon_fa3_forward.py --tune > ptx_dump
#1786649729
tpython gluon_attention_forward.py --tune
#1786661388
tpython benchmark.py
#1786662440
debugjob
#1786667375
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/4_partition_4096_128 python gluon_fa3_forward.py 
#1786667461
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/3_partition_4096_128 python gluon_attention_forward.py 
#1786672717
debugjob
#1786648404
load_module && start_gluon
#1786648407
cd ../attention/
#1786648415
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786648426
sq
#1786648898
scancel 761234
#1786650603
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786662188
sq
#1786662211
sq -i 60
#1786662418
scancel --me
#1786663134
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786663137
sq -i 60
#1786676498
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786676519
sq -i 60
#1786677550
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786677551
sq -i 60
#1786677577
scancel 762990
#1786677586
sq -i 60
#1786677955
scancel --me
#1786675033
sed -n '100,160p' attention/gluon_fa3_forward.py
#1786675038
sed -n '1,105p' attention/gluon_fa3_forward.py
#1786675092
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && tpython -c "import triton.language.core as c; import inspect; print(inspect.getsource(c._aggregate))"
#1786675108
cd /home/notming/links/scratch
#1786675110
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && tpython -c "import gluon_fa3_forward as g; print('Annotations:', getattr(g.PartitionArgs, '__annotations__', None))"
#1786675232
cd /home/notming/links/scratch
#1786675233
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && tpython -c "
#1786675233
class Foo:
#1786675233
    x: int
#1786675233
    y: str
#1786675233

#1786675233
print('Foo __annotations__:', Foo.__annotations__)
#1786675233
print('Foo dict keys:', Foo.__dict__.keys())
#1786675233
import annotationlib
#1786675233
print('annotationlib format_VALUE:', annotationlib.get_annotations(Foo))
#1786675233
"
#1786675256
cd /home/notming/links/scratch
#1786675257
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && tpython -c "import sys; print('Python sys.version:', sys.version)"
#1786675277
cd /home/notming/links/scratch
#1786675278
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && tpython gluon_fa3_forward.py
#1786675311
cd /home/notming/links/scratch
#1786675312
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && compute-sanitizer --tool memcheck tpython gluon_fa3_forward.py
#1786675325
cd /home/notming/links/scratch
#1786675326
source ~/.bashrc 2>/dev/null; load_module && start_gluon && which tpython && which compute-sanitizer
#1786675333
cd /home/notming/links/scratch
#1786675334
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && compute-sanitizer --tool memcheck --target-processes all apptainer exec --nvccli $SCRATCH/sparse.sif python gluon_fa3_forward.py
#1786675489
cd /home/notming/links/scratch
#1786675490
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675490
with open('gluon_attention_forward.py') as f:
#1786675490
    print('=== gluon_attention_forward.py ===')
#1786675490
    lines = f.readlines()
#1786675490
    print('Total lines:', len(lines))
#1786675490
    print(''.join(lines[:100]))
#1786675490
"
#1786675505
cd /home/notming/links/scratch
#1786675506
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675506
with open('gluon_attention_forward.py') as f:
#1786675506
    af_text = f.read()
#1786675506

#1786675506
with open('gluon_fa3_forward.py') as f:
#1786675506
    fa3_text = f.read()
#1786675506

#1786675506
print('=== PartitionArgs in gluon_attention_forward.py ===')
#1786675506
p1 = af_text.find('class PartitionArgs:')
#1786675506
p2 = af_text.find('@aggregate\nclass Counter:', p1)
#1786675506
print(af_text[p1:p2])
#1786675506

#1786675506
print('=== PartitionArgs in gluon_fa3_forward.py ===')
#1786675506
p1 = fa3_text.find('class PartitionArgs:')
#1786675506
p2 = fa3_text.find('@aggregate\nclass Counter:', p1)
#1786675506
print(fa3_text[p1:p2])
#1786675506
"
#1786675522
cd /home/notming/links/scratch
#1786675523
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675523
with open('gluon_attention_forward.py') as f:
#1786675523
    af_text = f.read()
#1786675523

#1786675523
with open('gluon_fa3_forward.py') as f:
#1786675523
    fa3_text = f.read()
#1786675523

#1786675523
import re
#1786675523
print('Functions in gluon_attention_forward.py:')
#1786675523
for m in re.finditer(r'(def \w+|class \w+|@\w+)', af_text):
#1786675523
    print(m.group(0))
#1786675523

#1786675523
print('\nFunctions in gluon_fa3_forward.py:')
#1786675523
for m in re.finditer(r'(def \w+|class \w+|@\w+)', fa3_text):
#1786675523
    print(m.group(0))
#1786675523
"
#1786675566
cd /home/notming/links/scratch
#1786675567
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && tpython gluon_attention_forward.py
#1786675611
cd /home/notming/links/scratch
#1786675613
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675613
with open('gluon_fa3_forward.py') as f:
#1786675613
    text = f.read()
#1786675613

#1786675613
# Let's inspect fa3_warp_specialized_kernel in gluon_fa3_forward.py
#1786675613
p = text.find('@gluon.jit\ndef fa3_warp_specialized_kernel')
#1786675613
print(text[p:])
#1786675613
"
#1786675628
cd /home/notming/links/scratch
#1786675629
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675629
with open('gluon_fa3_forward.py') as f:
#1786675629
    lines = f.readlines()
#1786675629
print(''.join(lines[200:400]))
#1786675629
"
#1786675642
cd /home/notming/links/scratch
#1786675644
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675644
with open('gluon_fa3_forward.py') as f:
#1786675644
    lines = f.readlines()
#1786675644
print(''.join(lines[400:600]))
#1786675644
"
#1786675656
cd /home/notming/links/scratch
#1786675657
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675657
with open('gluon_attention_forward.py') as f:
#1786675657
    text = f.read()
#1786675657

#1786675657
p = text.find('warp_specialize')
#1786675657
print(text[p-200:p+300])
#1786675657
"
#1786675670
cd /home/notming/links/scratch
#1786675671
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675671
with open('gluon_attention_forward.py') as f:
#1786675671
    text = f.read()
#1786675671

#1786675671
p = text.find('fa3_warp_specialized_kernel')
#1786675671
print(text[p:p+1200])
#1786675671
"
#1786675697
cd /home/notming/links/scratch
#1786675698
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675698
with open('gluon_attention_forward.py') as f:
#1786675698
    lines = f.readlines()
#1786675698
print(''.join(lines[450:520]))
#1786675698
"
#1786675715
cd /home/notming/links/scratch
#1786675716
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675716
with open('gluon_attention_forward.py') as f:
#1786675716
    lines = f.readlines()
#1786675716
print(''.join(lines[380:460]))
#1786675716
"
#1786675734
cd /home/notming/links/scratch
#1786675735
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675735
with open('gluon_fa3_forward.py') as f:
#1786675735
    text = f.read()
#1786675735

#1786675735
p1 = text.find('def fa3_producer_partition')
#1786675735
p2 = text.find('def fa3_consumer_wg0')
#1786675735
print(text[p1:p2])
#1786675735
"
#1786675751
cd /home/notming/links/scratch
#1786675752
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675752
with open('gluon_fa3_forward.py') as f:
#1786675752
    text = f.read()
#1786675752

#1786675752
p1 = text.find('def store_acc_to_smem_subtile')
#1786675752
p2 = text.find('def fa3_producer_partition')
#1786675752
print(text[p1:p2])
#1786675752
"
#1786675764
cd /home/notming/links/scratch
#1786675765
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675765
import inspect
#1786675765
from triton.experimental.gluon import language as gl
#1786675765
print(inspect.getsource(gl.warp_specialize))
#1786675765
"
#1786675781
cd /home/notming/links/scratch
#1786675782
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675782
with open('gluon_attention_forward.py') as f:
#1786675782
    text = f.read()
#1786675782

#1786675782
p1 = text.find('def store_acc_to_smem_subtile')
#1786675782
p2 = text.find('def fa3_producer_partition')
#1786675782
print(text[p1:p2])
#1786675782
"
#1786675795
cd /home/notming/links/scratch
#1786675796
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675796
with open('gluon_attention_forward.py') as f:
#1786675796
    text = f.read()
#1786675796

#1786675796
p1 = text.find('def fa3_store_partition')
#1786675796
p2 = text.find('def fa3_warp_specialized_kernel')
#1786675796
print(text[p1:p2])
#1786675796
"
#1786675812
cd /home/notming/links/scratch
#1786675813
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675813
import inspect
#1786675813
from triton.experimental.gluon import language as gl
#1786675813
print(inspect.signature(gl.warp_specialize))
#1786675813
"
#1786675831
cd /home/notming/links/scratch
#1786675832
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786675832
with open('gluon_fa3_forward.py') as f:
#1786675832
    code = f.read()
#1786675832

#1786675832
# Fix 1: mbarrier.init(kv_empty_bars.index(i), count=2)
#1786675832
code = code.replace('mbarrier.init(kv_empty_bars.index(i), count=1)', 'mbarrier.init(kv_empty_bars.index(i), count=2)')
#1786675832

#1786675832
# Fix 2: worker_num_regs in warp_specialize
#1786675832
code = code.replace('], [num_warps, 1, 1])', '], [num_warps, 1, 1], [128, 24, 24])')
#1786675832

#1786675832
with open('gluon_fa3_forward_test.py', 'w') as f:
#1786675832
    f.write(code)
#1786675832

#1786675832
print('Saved gluon_fa3_forward_test.py')
#1786675832
"
#1786675852
cd /home/notming/links/scratch
#1786675853
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && tpython gluon_fa3_forward_test.py
#1786675889
cd /home/notming/links/scratch
#1786675890
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && tpython gluon_fa3_forward_test.py --tune
#1786676012
cd /home/notming/links/scratch
#1786676013
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && diff -u gluon_fa3_forward.py gluon_fa3_forward_test.py
#1786676022
cd /home/notming/links/scratch
#1786676023
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && cp gluon_fa3_forward_test.py gluon_fa3_forward.py && rm gluon_fa3_forward_test.py
#1786676093
cd /home/notming/links/scratch
#1786676094
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && tpython gluon_fa3_forward.py
#1786677206
cd /home/notming/links/scratch
#1786677208
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786677208
with open('gluon_fa3_forward.py') as f:
#1786677208
    text = f.read()
#1786677208

#1786677208
p1 = text.find('def fa3_get_configs')
#1786677208
p2 = text.find('def run_fa3_kernel')
#1786677208
print('=== gluon_fa3_forward.py ===')
#1786677208
print(text[p1:p2])
#1786677208
"
#1786677232
cd /home/notming/links/scratch
#1786677234
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786677234
with open('gluon_attention_forward.py') as f:
#1786677234
    text = f.read()
#1786677234

#1786677234
p1 = text.find('def fa3_get_configs')
#1786677234
p2 = text.find('def run_fa3_kernel')
#1786677234
print('=== gluon_attention_forward.py ===')
#1786677234
print(text[p1:p2])
#1786677234
"
#1786677251
cd /home/notming/links/scratch
#1786677252
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && python3 -c "
#1786677252
with open('gluon_fa3_forward.py') as f:
#1786677252
    code = f.read()
#1786677252

#1786677252
# Fix 1: 356 typo -> 256
#1786677252
code = code.replace('for BN in (64, 128, 356)', 'for BN in (64, 128, 256)')
#1786677252

#1786677252
# Fix 2: worker_num_regs in warp_specialize
#1786677252
code = code.replace('], [num_warps, 1, 1])', '], [num_warps, 1, 1], [128, 24, 24])')
#1786677252

#1786677252
# Also check SF in configs: (1, 2, 4, 8)
#1786677252
code = code.replace('for SF in (4, 8)', 'for SF in (1, 2, 4, 8)')
#1786677252

#1786677252
with open('gluon_fa3_forward.py', 'w') as f:
#1786677252
    f.write(code)
#1786677252

#1786677252
print('Updated gluon_fa3_forward.py')
#1786677252
"
#1786677267
cd /home/notming/links/scratch
#1786677268
source ~/.bashrc 2>/dev/null; load_module && start_gluon && cd ../attention && tpython gluon_fa3_forward.py --tune
#1786723232
load_module && start_gluon && cd ../att
#1786723241
cd ../attention/
#1786723347
tpython gluon_fa3_forward.py --tune
#1786723522
tpython gluon_fa3_forward.py --tune > ptx_dump.txt
#1786723596
tpython gluon_fa3_forward.py --tune > ptx_dump
#1786724117
tpython benchmark.py --head-dim=64
#1786751144
load_module && start_gluon && cd ../attention
#1786751155
tpython -c "import os; os.system('compute-sanitizer --tool=racecheck python gluon_fa3_forward.py')"
#1786751198
tpython -c "import os; os.system('compute-sanitizer --tool=memcheck python gluon_fa3_forward.py')"
#1786751304
tpython -c "import os; os.system('compute-sanitizer --tool=racecheck python gluon_fa3_forward.py')"
#1786762789
load_module && start_gluon && cd ../attention
#1786762812
tpython gluon_3_partition_pingpong.py
#1786762889
tpython benchmark.py --head-dim=128
#1786764057
tpython gluon_3_partition_pingpong.py
#1786764262
tpython benchmark.py --head-dim=128
#1786765151
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/3_partition_pingpong_4096_128 python gluon_3_partition_pingpong.py 
#1786765612
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/3_partition_pingpong_4096_128 python gluon_3_partition_pingpong.py --bm 128 --bn 128 --stages 4 --sf 2 --warps 8
#1786765780
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/3_partition_pingpong_4096_128 python gluon_3_partition_pingpong.py --bm 128 --bn 256 --stages 2 --sf 2 --warps 8
#1786765800
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/3_partition_pingpong_4096_128 python gluon_3_partition_pingpong.py --bm 128 --bn 128 --stages 2 --sf 2 --warps 8
#1786765926
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/3_partition_pingpong_4096_128 python gluon_3_partition_pingpong.py 
#1786765984
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/3_partition_pingpong_suboptimal_4096_128 python gluon_3_partition_pingpong.py --bm 128 --bn 128 --stages 2 --sf 2 --warps 8
#1786731374
grep -E "(load_module|start_gluon|tpython)" ~/.bashrc
#1786724587
load_module && start_gluon && cd ../attention
#1786724608
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786724619
sq
#1786746656
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786746658
sq
#1786748275
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786748289
scancel 769802
#1786749624
sq
#1786750249
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786750251
sq
#1786760445
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786760452
sq
#1786760984
scancel --me
#1786761010
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786761013
sq
#1786762022
scancel --me
#1786762073
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786762758
sq
#1786762766
scancel 771253
#1786723206
debugjob
#1786728433
load_module && start_gluon && cd ../attention
#1786728502
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/4_partition_4096_128 python gluon_fa3_forward.py 
#1786729002
tpython gluon_fa3_forward.py 
#1786731492
debugjob
#1786745591
tpython gluon_fa3_forward.py 
#1786746329
tpython gluon_fa3_forward.py --tune
#1786747923
tpython gluon_fa3_forward.py 
#1786748000
tpython gluon_fa3_forward.py --tune
#1786750493
tpython gluon_fa3_forward.py 
#1786750523
compute-sanitizer --tool=synccheck tpython gluon_fa3_forward.py 
#1786750539
compute-sanitizer tpython gluon_fa3_forward.py 
#1786750655
tpython -c "import os; os.system('compute-sanitizer --tool=synccheck python gluon_fa3_forward.py')"
#1786750733
tpython -c "import os; os.system('compute-sanitizer --tool=racecheck python gluon_fa3_forward.py')"
#1786751117
debugjob
#1786757004
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/4_partition_4096_128 python gluon_fa3_forward.py 
#1786758692
tpython gluon_fa3_forward.py 
#1786762749
debugjob
#1786730905
head -n 50 /home/notming/links/scratch/attention/gluon_fa3_forward.py
#1786730912
nl -ba /home/notming/links/scratch/attention/gluon_fa3_forward.py | sed -n '50,220p'
#1786730915
nl -ba /home/notming/links/scratch/attention/gluon_fa3_forward.py | sed -n '220,380p'
#1786730920
nl -ba /home/notming/links/scratch/attention/gluon_fa3_forward.py | sed -n '370,550p'
#1786730923
nl -ba /home/notming/links/scratch/attention/gluon_fa3_forward.py | sed -n '550,750p'
#1786730927
nl -ba /home/notming/links/scratch/attention/gluon_fa3_forward.py | sed -n '750,860p'
#1786730932
cat /home/notming/links/scratch/attention/common.py
#1786730941
grep -rn "warp_specialize" /home/notming/links/scratch
#1786730950
ls -la /home/notming/links/scratch/attention
#1786730957
head -n 100 /home/notming/links/scratch/attention/gluon_attention_forward.py
#1786730972
nl -ba /home/notming/links/scratch/attention/gluon_attention_forward.py | sed -n '380,500p'
#1786730992
grep -n "\* 2" /home/notming/links/scratch/attention/gluon_fa3_forward.py
#1786731003
sed -n '411,485p' /home/notming/links/scratch/attention/gluon_attention_forward.py
#1786731046
sed -n '310,370p' /home/notming/links/scratch/attention/gluon_attention_forward.py
#1786731053
sed -n '380,495p' /home/notming/links/scratch/attention/gluon_fa3_forward.py
#1786731076
python3 -c '
#1786731076
path = "/home/notming/links/scratch/attention/gluon_fa3_forward.py"
#1786731076
with open(path, "r") as f:
#1786731076
    content = f.read()
#1786731076

#1786731076
# 1. Fix scheduler initialize * 2
#1786731076
old1 = "scheduler = SchedulerImpl.initialize(p.o0_desc.shape[0] * 2, p.o0_desc.shape[1], BLOCK_M, BLOCK_K)"
#1786731076
new1 = "scheduler = SchedulerImpl.initialize(p.o0_desc.shape[0], p.o0_desc.shape[1], BLOCK_M, BLOCK_K)"
#1786731076
old2 = "scheduler = SchedulerImpl.initialize(p.o1_desc.shape[0] * 2, p.o1_desc.shape[1], BLOCK_M, BLOCK_K)"
#1786731076
new2 = "scheduler = SchedulerImpl.initialize(p.o1_desc.shape[0], p.o1_desc.shape[1], BLOCK_M, BLOCK_K)"
#1786731076

#1786731076
content = content.replace(old1, new1).replace(old2, new2)
#1786731076

#1786731076
# 2. Fix valid() in fa3_get_configs
#1786731076
old_valid_bm = "        if BM < 32 or BM % 2 != 0:\n            return False"
#1786731076
new_valid_bm = "        SUB_BM = BM // 2\n        if SUB_BM % 64 != 0:\n            return False"
#1786731076
content = content.replace(old_valid_bm, new_valid_bm)
#1786731076

#1786731076
old_split_k = "        split_k = BK // SF\n        if split_k < 16:\n            return False"
#1786731076
new_split_k = "        split_k = BK // SF\n        if split_k < 32:\n            return False"
#1786731076
content = content.replace(old_split_k, new_split_k)
#1786731076

#1786731076
old_for_bm = "        for BM in (64, 128, 256)"
#1786731076
new_for_bm = "        for BM in (128, 256)"
#1786731076
content = content.replace(old_for_bm, new_for_bm)
#1786731076

#1786731076
# 3. Fix MLIR dump directory creation
#1786731076
old_dump = "    os.environ[\"MLIR_ENABLE_DUMP\"]=\"1\"\n    os.environ[\"MLIR_DUMP_PATH\"] = \"./MLIR_DUMP/4_partition_4096_128\""
#1786731076
new_dump = "    os.environ[\"MLIR_ENABLE_DUMP\"]=\"1\"\n    os.environ[\"MLIR_DUMP_PATH\"] = \"./MLIR_DUMP/4_partition_4096_128\"\n    os.makedirs(os.environ[\"MLIR_DUMP_PATH\"], exist_ok=True)"
#1786731076
content = content.replace(old_dump, new_dump)
#1786731076

#1786731076
with open(path, "w") as f:
#1786731076
    f.write(content)
#1786731076
print("Updated successfully")
#1786731076
'
#1786731089
cd /home/notming/links/scratch/attention && apptainer exec --nvccli $SCRATCH/sparse.sif python gluon_fa3_forward.py
#1786731108
cd /home/notming/links/scratch
#1786731109
apptainer exec --nvccli /scratch/notming/sparse.sif python /scratch/notming/attention/gluon_fa3_forward.py
#1786731157
bash -i -c "load_module && start_gluon && cd /scratch/notming/attention && tpython gluon_fa3_forward.py"
#1786731195
python3 -c '
#1786731195
path = "/home/notming/links/scratch/attention/gluon_fa3_forward.py"
#1786731195
with open(path, "r") as f:
#1786731195
    content = f.read()
#1786731195

#1786731195
old_dump = "    os.makedirs(os.environ[\"MLIR_DUMP_PATH\"], exist_ok=True)"
#1786731195
new_dump = "    os.makedirs(os.path.dirname(os.environ[\"MLIR_DUMP_PATH\"]), exist_ok=True)"
#1786731195
content = content.replace(old_dump, new_dump)
#1786731195

#1786731195
with open(path, "w") as f:
#1786731195
    f.write(content)
#1786731195
print("Updated dump path")
#1786731195
'
#1786731206
bash -i -c "load_module && start_gluon && cd /scratch/notming/attention && tpython gluon_fa3_forward.py"
#1786731242
sed -n '150,310p' /home/notming/links/scratch/attention/gluon_attention_forward.py
#1786731250
grep -n "Counter.create" /home/notming/links/scratch/attention/gluon_fa3_forward.py
#1786731265
python3 -c '
#1786731265
path = "/home/notming/links/scratch/attention/gluon_fa3_forward.py"
#1786731265
with open(path, "r") as f:
#1786731265
    content = f.read()
#1786731265

#1786731265
# 1. Fix producer Counter.create(0, ...) -> Counter.create(1, ...)
#1786731265
old_prod = """    kv_state = Counter.create(0, p.kv_empty_bars.shape[0])
#1786731265
    q_state = Counter.create(0, p.q_empty_bar.shape[0])"""
#1786731265
new_prod = """    kv_state = Counter.create(1, p.kv_empty_bars.shape[0])
#1786731265
    q_state = Counter.create(1, p.q_empty_bar.shape[0])"""
#1786731265
assert old_prod in content, "old_prod not found"
#1786731265
content = content.replace(old_prod, new_prod, 1)
#1786731265

#1786731265
# 2. Fix consumer wg0
#1786731265
old_wg0_outer = """    acc_state = Counter.create(0, p.o0_empty_bars.shape[0])
#1786731265
    q_state = Counter.create(0, p.q_empty_bar.shape[0])
#1786731265
    
#1786731265
    num_steps = SEQ_LEN // BLOCK_N"""
#1786731265
new_wg0_outer = """    acc_state = Counter.create(1, p.o0_empty_bars.shape[0])
#1786731265
    q_state = Counter.create(0, p.q_empty_bar.shape[0])
#1786731265
    k_state = Counter.create(0, num_stages)
#1786731265
    v_state = Counter.create(0, num_stages)
#1786731265
    
#1786731265
    num_steps = SEQ_LEN // BLOCK_N"""
#1786731265
assert old_wg0_outer in content, "old_wg0_outer not found"
#1786731265
content = content.replace(old_wg0_outer, new_wg0_outer, 1)
#1786731265

#1786731265
old_wg0_inner = """        m_old = gl.full((SUB_BM,), -float(\x27inf\x27), dtype=gl.float32, layout=s_layout)
#1786731265
        l_old = gl.zeros((SUB_BM,), dtype=gl.float32, layout=s_layout)
#1786731265
        
#1786731265
        k_state = Counter.create(0, num_stages)
#1786731265
        v_state = Counter.create(0, num_stages)
#1786731265

#1786731265
        mbarrier.wait(p.q_ready_bar.index(0), q_state.phase)"""
#1786731265
new_wg0_inner = """        m_old = gl.full((SUB_BM,), -float(\x27inf\x27), dtype=gl.float32, layout=s_layout)
#1786731265
        l_old = gl.zeros((SUB_BM,), dtype=gl.float32, layout=s_layout)
#1786731265

#1786731265
        mbarrier.wait(p.q_ready_bar.index(0), q_state.phase)"""
#1786731265
assert old_wg0_inner in content, "old_wg0_inner not found"
#1786731265
content = content.replace(old_wg0_inner, new_wg0_inner, 1)
#1786731265

#1786731265
# 3. Fix consumer wg1
#1786731265
old_wg1_outer = """    acc_state = Counter.create(0, p.o1_empty_bars.shape[0])
#1786731265
    q_state = Counter.create(0, p.q_empty_bar.shape[0])
#1786731265
    
#1786731265
    num_steps = SEQ_LEN // BLOCK_N"""
#1786731265
new_wg1_outer = """    acc_state = Counter.create(1, p.o1_empty_bars.shape[0])
#1786731265
    q_state = Counter.create(0, p.q_empty_bar.shape[0])
#1786731265
    k_state = Counter.create(0, num_stages)
#1786731265
    v_state = Counter.create(0, num_stages)
#1786731265
    
#1786731265
    num_steps = SEQ_LEN // BLOCK_N"""
#1786731265
assert old_wg1_outer in content, "old_wg1_outer not found"
#1786731265
content = content.replace(old_wg1_outer, new_wg1_outer, 1)
#1786731265

#1786731265
old_wg1_inner = """        m_old = gl.full((SUB_BM,), -float(\x27inf\x27), dtype=gl.float32, layout=s_layout)
#1786731265
        l_old = gl.zeros((SUB_BM,), dtype=gl.float32, layout=s_layout)
#1786731265
        
#1786731265
        # Track K prefetching vs V consumption/release separately
#1786731265
        k_state = Counter.create(0, num_stages)
#1786731265
        v_state = Counter.create(0, num_stages)
#1786731265

#1786731265
        mbarrier.wait(p.q_ready_bar.index(0), q_state.phase)"""
#1786731265
new_wg1_inner = """        m_old = gl.full((SUB_BM,), -float(\x27inf\x27), dtype=gl.float32, layout=s_layout)
#1786731265
        l_old = gl.zeros((SUB_BM,), dtype=gl.float32, layout=s_layout)
#1786731265

#1786731265
        mbarrier.wait(p.q_ready_bar.index(0), q_state.phase)"""
#1786731265
assert old_wg1_inner in content, "old_wg1_inner not found"
#1786731265
content = content.replace(old_wg1_inner, new_wg1_inner, 1)
#1786731265

#1786731265
with open(path, "w") as f:
#1786731265
    f.write(content)
#1786731265
print("Updated Counter phases successfully!")
#1786731265
'
#1786731279
bash -i -c "load_module && start_gluon && cd /scratch/notming/attention && tpython gluon_fa3_forward.py"
#1786863033
load_module && start_gluon
#1786863037
cd ../attention/
#1786863049
tpython gluon_attention_forward.py 
#1786863079
nvidia-smi
#1786863084
kill -9 1405182
#1786863244
tpython gluon_attention_forward.py 
#1786863438
nvidia-smi
#1786863445
kill -9 1406648
#1786863450
tpython gluon_attention_forward.py > test.txt
#1786863598
nvidia-smi
#1786863607
kill -9 1407261 1407835
#1786863925
tpython gluon_attention_forward.py > test.txt
#1786863975
nvidia-smi
#1786863982
kill -9 1409641
#1786863984
tpython gluon_attention_forward.py > test.txt
#1786864027
nvidia-smi
#1786864032

#1786864042
tpython gluon_attention_forward.py > test.txt
#1786864048
nvidia-smi
#1786864058
tpython gluon_attention_forward.py
#1786864276
nvidia-smi
#1786864284
kill -9 1410317
#1786864291
tpython gluon_attention_forward.py 
#1786864384
nvidia-smi
#1786864388
kill -9 1412020
#1786864393
tpython gluon_attention_forward.py 
#1786832862
load_module && start_gluon && cd ../attention
#1786832953
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/3_partition_pingpong_4096_128.ncu-rep python gluon_3_partition_pingpong.py 
#1786833104
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786833708
sq
#1786839966
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/3_partition_pingpong_4096_128.ncu-rep python gluon_3_partition_pingpong.py 
#1786856424
tpython gluon_3_partition_pingpong.py --tune > ptx_dump 
#1786856639
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786856674
scancel --me
#1786856684
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786858731
scancel --me
#1786858820
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/3_partition_suboptimal_4096_128.ncu-rep python gluon_attention_forward.py 
#1786859666
tpython benchmark.py 
#1786860368
nvidia-smi
#1786860375
tpython benchmark.py 
#1786860511
tpython gluon_attention_forward.py --tune
#1786860596
tpython benchmark.py 
#1786861475
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/3_partition_4096_128.ncu-rep python gluon_attention_forward.py 
#1786861900
tpython gluon_attention_forward.py
#1786862033
tpython benchmark.py 
#1786862604
tpython gluon_attention_forward.py
#1786862638
nvidia-smi
#1786862659
kill -9 2990853 3247101
#1786862996
debugjob
#1786892151
load_module && start_gluon && cd ../attention
#1786892163
tpython gluon_attention_forward.py 
#1786892265
tpython benchmark.py 
#1786892878
tpython benchmark.py --skip-4part
#1786892934
[A
#1786892935
tpython benchmark.py --skip-4part
#1786932462
load_module && start_gluon && cd ../attention
#1786932477
tpython gluon_3_partition_pingpong.py 
#1786932888
nvidia-smi
#1786932896
kill -9 1658480 1659066
#1786932900
tpython gluon_3_partition_pingpong.py 
#1786935264
ptxas -v --gpu-name=sm_90a compiler_scratch/triton_cache_1674287/KOOEU547QBPC2PKXTOS6YVIBFBNHP4RNCCH5VIV7W2V2TNCWEG6Q/fa3_warp_specialized_kernel.ptx -o /dev/null
#1786891685
load_module && start_gluon && cd ../attention
#1786891703
tpython gluon_attention_forward.py 
#1786891738
tpython benchmark.py 
#1786892132
debugjob
#1786904409
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o "Profiling/3_partition_4096_128" python gluon_attention_forward.py 
#1786904526
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o "Profiling/3_partition_pingpong_4096_128" python gluon_3_partition_pingpong.py 
#1786905364
tpython gluon_attention_forward.py 
#1786906609
tpython benchmark.py --skip-4part
#1786907645
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o "Profiling/3_partition_pingpong_suboptimal_4096_128" python gluon_3_partition_pingpong.py 
#1786908098
tpython benchmark.py --skip-4part
#1786918904
tpython gluon_3_partition_pingpong.py 
#1786918983
tpython benchmark.py --skip-4part
#1786920199
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o "Profiling/3_partition_pingpong_suboptimal_4096_128" python gluon_3_partition_pingpong.py 
#1786920610
tpython gluon_3_partition_pingpong.py 
#1786920682
tpython benchmark.py --skip-4part
#1786921491
tpython gluon_3_partition_pingpong.py 
#1786921535
tpython benchmark.py --skip-4part
#1786921712
tpython gluon_3_partition_pingpong.py 
#1786921734
tpython benchmark.py --skip-4part
#1786924427
tpython gluon_3_partition_pingpong.py 
#1786924497
tpython benchmark.py --skip-4part
#1786924906
tpython gluon_3_partition_pingpong.py 
#1786925004
tpython benchmark.py --skip-4part
#1786930982
tpython gluon_3_partition_pingpong.py 
#1786893548
load_module && start_gluon && cd ../attention
#1786893558
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786895081
sq
#1786896114
scancel 782899
#1786897111
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786897120
sq
#1786922668
apptainer exec $SCRATCH/sparse.sif --nvccli ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/3_partition_pingpong_suboptimal_4096_128 python gluon_3_partition_pingpong.py 
#1786922685
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/3_partition_pingpong_suboptimal_4096_128 python gluon_3_partition_pingpong.py 
#1786925493
ptxas -v --gpu-name=sm_90a compiler_scratch/triton_cache_3958640/I3Z2NR23ALPE3K4GVH5WHQ4XMPRGNC2S5T5JSHJRKCNX2CQ6YZAQ/fa3_warp_specialized_kernel.ptx -o /dev/null
#1786925617
ptxas -v --gpu-name=sm_90a --maxrregcount=256 compiler_scratch/triton_cache_3958640/I3Z2NR23ALPE3K4GVH5WHQ4XMPRGNC2S5T5JSHJRKCNX2CQ6YZAQ/fa3_warp_specialized_kernel.ptx -o /dev/null
#1786925883
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/3_partition_pingpong_4096_128 python gluon_3_partition_pingpong.py 
#1786925937
apptainer exec --nvccli $SCRATCH/sparse.sif ncu --set full -f -k "fa3_warp_specialized_kernel" -o Profiling/3_partition_4096_128 python gluon_attention_forward.py 
#1786927380
sbatch sbatch_sh/benchmark_fa3_baseline.sh 
#1786927382
sq
#1786931719
ptxas -v --gpu-name=sm_90a compiler_scratch/triton_cache_1498481/GPUMBIOAF3KQL7IUUFRCHJWMWXM5W5XJW7YTFHRRB54HI64PI5AA/fa3_warp_specialized_kernel.ptx -o /dev/null
#1786932444
debugjob
#1786981135
load_module && start_gluon && cd ../attention
#1786981144
tpython gluon_3_partition_pingpong.py 
#1786982599
tpython gluon_3_partition_pingpong.py > test.txt
#1786982683
tpython gluon_3_partition_pingpong.py 
#1786983873
ptxas -v --gpu-name=sm_90a compiler_scratch/triton_cache_66508/EEJ5CFVLM5J75VA5UNE4WLNMO23YG3MGPSL5Z36U57ITIL3I74TQ/fa3_warp_specialized_kernel.ptx -o /dev/null
#1786983902
tpython gluon_3_partition_pingpong.py --tune
#1786984157
tpython gluon_attention_forward.py --tune
#1786984240
tpython gluon_3_partition_pingpong.py --tune
#1786984421
tpython gluon_3_partition_pingpong.py
#1786985112
tpython gluon_3_partition_pingpong.py --tune
