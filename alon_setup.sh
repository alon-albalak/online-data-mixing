conda activate base
conda create --name gptneox python=3.8
conda activate gptneox
conda install pytorch pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-flashattention.txt
python ./megatron/fused_kernels/setup.py install

sed -i 's/from torch._six import inf/from torch import inf/g' ${CONDA_PREFIX}/lib/python3.8/site-packages/deepspeed/runtime/utils.py
sed -i 's/from torch._six import inf/from torch import inf/g' ${CONDA_PREFIX}/lib/python3.8/site-packages/deepspeed/runtime/zero/stage2.py
sed -i 's/from torch._six import inf/from torch import inf/g' ${CONDA_PREFIX}/lib/python3.8/site-packages/deepspeed/runtime/zero/stage3.py
mkdir outputs