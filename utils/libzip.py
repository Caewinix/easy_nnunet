from typing import Optional
import os
import shutil
from zipfile import ZipFile


class ZipDirname(str): ...
class ZipFilename(str): ...


def _get_zip_filelist_with_dirname(zip_file: ZipFile, dirname):
    for file in zip_file.filelist:
        filename = file.filename
        if filename.startswith(dirname):
            filename = filename[len(dirname):]
            if filename == '/':
                continue
            else:
                filename = filename[1:]
        else:
            continue
        yield file.is_dir(), filename


def iterdir(zip_file: ZipFile, dirname: str = ''):
    if dirname is None or dirname == '':
        filelist_iter = ((file.is_dir(), file.filename) for file in zip_file.filelist)
    else:
        filelist_iter = _get_zip_filelist_with_dirname(zip_file, dirname)
    for is_dir, filename in filelist_iter:
        if is_dir:
            yield ZipDirname(filename[:-1])
        elif '/' not in filename:
            yield ZipFilename(filename)


def listdir(zip_file: ZipFile, dirname: str = ''):
    return list(iterdir(zip_file, dirname))


def extract(zip_file: ZipFile, src: str, dst: str, password: Optional[str] = None):
    with zip_file.open(src, pwd=password) as source, open(dst, "wb") as target:
        shutil.copyfileobj(source, target)


def extract_to(zip_file: ZipFile, src: str, dir: str, password: Optional[str] = None):
    dst = os.path.join(dir, os.path.basename(src))
    extract(zip_file, src, dst, password)