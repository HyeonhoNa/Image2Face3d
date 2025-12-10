conda env create -f environment.yml
git clone --recurse-submodules https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
pip install ./diff-gaussian-rasterization

python example.py