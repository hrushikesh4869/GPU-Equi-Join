
#read csv file as a dictionary
import csv

dict_sid = {}
dict_rid = {}

with open('table_s.csv', mode='r') as infile:
    reader = csv.reader(infile)
    dict_sid = {rows[0]:rows[1] for rows in reader}

with open('table_r.csv', mode='r') as infile:
    reader = csv.reader(infile)
    dict_rid = {rows[0]:rows[1] for rows in reader}




with open("output",'r') as f:
    lines = f.readlines()
    for line in lines:
        if "SID" in line:
            line = line.split(" ")
            sid = line[7].split("=")[1][:-1]
            rid = line[8].split("=")[1][:-1]
            if dict_sid[sid] != dict_rid[rid]:
                print("SID: " + sid + " RID: " + rid + " is not correct")            