pkill -f python3.12
pkill -f tensorboard

# wait 2 seconds
sleep 2

rm -rf out.log tb.log
nohup python3.12 -u train.py >> out.log 2>&1 &
nohup tensorboard --logdir logs --port 6006 --bind_all >> tb.log 2>&1 &
tail -f out.log
