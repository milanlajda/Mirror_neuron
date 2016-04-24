from JOY import joy
# from SADNESS import sadness
from ANGER import anger
from FEAR import fear


# function that feeds the outside nontraining data to all the emotion networks
def feedthedata(outsidedata):
    joyvariable = joy(outsidedata)
    fearvariable = fear(outsidedata)
    angervariable = anger(outsidedata)

    if joyvariable > 0.7:
        print("It's JOY!")
    elif fearvariable > 0.7:
        print("It's FEAR!")
    elif angervariable > 0.7:
        print("It' ANGER!")
    else:
        print("ERROR!")
