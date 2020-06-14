#! python
import sys
import glob
from pathlib import Path
from random import shuffle, seed
import shutil
from os.path import basename
seed(4)

def copy_files(files: list, dir: Path) -> None:
    for file in files:
        shutil.copyfile(str(file), str(dir/basename(file)))

dir, ratio, train_dir, test_dir = sys.argv[1:]
dir = Path(dir)
ratio = float(ratio)
assert 0 <= ratio <= 1.0
train_dir = Path(train_dir)
if train_dir.exists():
    shutil.rmtree(str(train_dir))
train_dir.mkdir()
test_dir = Path(test_dir)
if test_dir.exists():
    shutil.rmtree(str(test_dir))
test_dir.mkdir()

all_files = list(dir.glob('*jpg'))
assert len(all_files) > 0

shuffle(all_files)
test_files_count = int(ratio * len(all_files))
test_files = all_files[:test_files_count]
copy_files(test_files, test_dir)

train_files = all_files[test_files_count:]
copy_files(train_files, train_dir)



