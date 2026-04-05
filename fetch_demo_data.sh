pip install gdown
cd "$(dirname "$0")"
gdown --folder "https://drive.google.com/drive/folders/1-S8oZa3TWx41bblIub32L6aqkgp3p3oj?usp=sharing"

if [ -d "rants_demo_data/_DATA" ]; then
    rm -rf ./_DATA
    mv rants_demo_data/_DATA ./_DATA
fi
