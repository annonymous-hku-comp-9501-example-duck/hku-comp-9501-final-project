
root = '/home/geiger/krenz73/coding/06_drivelm/DriveLM/outputs/01_DriveLMv2_RT2'

import os
import sys

# find all files in root directory with .ckpt extension
for root, dirs, files in os.walk(root):
    for file in files:
        if file.endswith('.ckpt'):
            if 'last' in file:
                continue
            # names:epoch=001.ckpt
            # print(file)
            # if int(file.split('=')[1].split('.')[0]) == 1:
            #     continue
            # elif int(file.split('=')[1].split('.')[0]) == 3:
            #     continue
            # if int(file.split('=')[1].split('.')[0]) == 29:
            #     continue
            # delete all but not 5, 10, 15, 20 with modulo
            try:
                if int(file.split('=')[1].split('.')[0]) % 5 == 0:
                    continue
            except:
                pass
            else:
                print(os.path.join(root, file))
                # remove with "": > file" and then "rm file"
                # os.system(': > ' + os.path.join(root, file))
                os.system('rm ' + os.path.join(root, file))


                # os.remove(os.path.join(root, file))
