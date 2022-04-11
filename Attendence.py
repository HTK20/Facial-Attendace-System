import record
import train
import test

print("enter your choice \n1. New User \n2. Existing User")
ch = input()

if ch=="1":
    record.face_record()
    train.train_model()
    mar = input("Want to mark your attendence ? (y\n)")
    if mar=="y" or mar=="Y":
        test.test_model()
    else:
        exit()

elif ch=="2":
    test.test_model()
