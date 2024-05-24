size=4

for((rank=0;rank<size;rank++))
do
	screen_name="fe"$rank
	screen -dmS $screen_name
	screen -x -S $screen_name -p 0 -X stuff "python3 test.py --size="$size" --rank="$rank" | tee .log"$rank"\n"
done


rm *checkpoint.json
rm *fail_list.json

size=4
for((rank=0;rank<size;rank++))
do
	screen_name="fe"$rank
	screen -S $screen_name -X quit
done
