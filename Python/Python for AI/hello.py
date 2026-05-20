name="amrit"
string=f"my Name is {name}"

print(string.upper())

temperature=25

if temperature>20:
    print("hot")
else:
    print("cold")

for i in name:
    print(i)

import datetime


today=datetime.datetime.now()
print(today)


try:
    result=12/0
except:
    print("DIvision rule error!")
