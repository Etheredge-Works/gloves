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
for file in all_files:
    file_name = str(file)
    if "Maine_Coon" in file_name or "boxer" in file_name:
        shutil.copyfile(file_name, str(test_dir / basename(file)))
    else:
        shutil.copyfile(file_name, str(train_dir / basename(file)))

#all_main_coon_files = list(dir.glob("Main_Coon*"))
#all_boxer_files = list(dir.glob("boxer*"))


#shuffle(all_files)
#test_files_count = int(ratio * len(all_files))
#test_file_count = len(all_main_coon_files) + len(all_boxer_files)
#test_files = all_files[:test_files_count]
#copy_files(all_main_coon_files, test_dir)
#copy_files(all_boxer_files, test_dir)


#train_files = all_files[test_files_count:]
#copy_files(train_files, train_dir)



