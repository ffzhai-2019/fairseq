var=`ps aux | grep "train_parallel" | grep "sh" | awk '{print $2}'`
echo $var
kill -9 $var
ps aux | grep "python" | grep "train" | awk '{print $2}' | xargs kill -9
#ps aux | grep "python" | grep "multiprocessing" | awk '{print $2}' | xargs kill -9
