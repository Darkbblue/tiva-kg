# 避免非法操作
if [ $# -ne 1 ]; then
	echo "usage: ./decompress <prefix_of_tar>"
	exit
fi

# 合并分卷并解压缩
cat raw/$1.tar* > raw/$1.tar.gz
tar xzvf raw/$1.tar.gz

# 文件夹内容合并
cp -r $1/* m

# 清理
rm -rf raw/$1* $1*

