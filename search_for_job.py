import glob

path = 'outputs/01_DriveLMv2_RT2'
jobnr = 95387

#search for all qsub_out1.log files
for file in glob.glob(path + '/**/qsub_out*.log', recursive=True):
    # somewhere in the file: JobId=95407 JobName=drivelm_inp_1
    # check if JobID = jobnr
    with open(file, 'r') as f:
        content = f.read()
        if f'JobId={jobnr}' in content:
            print('found\n')
            print(file)