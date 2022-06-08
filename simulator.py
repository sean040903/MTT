import numpy as np

#effect
deathgard = 0
reburth = 0
turn = 0
atk = 0
health = 0
attackaftergard = 0
damage = 0
energy = 0
highesthealth = 0


#skill function
def deathgard(): # 루시우스: if reburth += 1, 발제: 일반공격전, 바바라: 일반공격전, 세실리아: 전투시작시(4턴지속), 아누비스: 전투시작시(30턴 지속)
    if deathgard == 0:
        deathgard = 유효일반공격횟수
    else:
        pass

def huntinginstinct(atk, attackaftergard): #유리: 2회, 50%증가, 전투시작전
    if attackaftergard == 2:
        atk = atk*(2/3)
    elif attackaftergard == 0:
        atk = atk*1.5
    return atk, attackaftergard

def stoneskin(deatgard):# 루시우스 highdeathgard
    pass

def energygard(): #마모니르, 옥토들의왕
    if turn == 0:
        energy = health*1.5
    else:
        energy += highesthealth*0.1
