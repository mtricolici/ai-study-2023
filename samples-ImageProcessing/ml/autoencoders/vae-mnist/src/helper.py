from datetime import datetime
  
#########################################################
def lm(msg, end='\n'):
    tm = datetime.now().strftime("%H:%M:%S")
    print(f'{tm} {msg}', end=end)
#########################################################

