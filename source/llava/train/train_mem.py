# from llava.train.train import train



import os 
os.environ["WANDB_API_KEY"] = 'bcd2d5eb473ab8f9d49469b13d8c457c80087b98' # 将引号内的+替换成自己在wandb上的一串值
os.environ["WANDB_MODE"] = "offline"   # 离线  （此行代码不用修改）


from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
