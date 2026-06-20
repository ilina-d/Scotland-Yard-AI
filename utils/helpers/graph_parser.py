data = ''

for line in data.split('\n'):
    if not line or line.startswith('#'):
        continue

    line = line.replace(' ', '').split(';')
    leng = len(line)

    # START NODE; TAXI END NODES; BUS END NODES; METRO END NODES; FERRY END NODES
    sn = line[0]

    if leng >= 2 and line[1]:  # has taxi
        for en in line[1].split(','):
            if en:
                print(f'{sn},{en},taxi')

    if leng >= 3 and line[2]:  # has bus
        for en in line[2].split(','):
            if en:
                print(f'{sn},{en},bus')

    if leng >= 4 and line[3]:  # has metro
        for en in line[3].split(','):
            if en:
                print(f'{sn},{en},metro')

    if leng >= 5 and line[4]:  # has ferry
        for en in line[4].split(','):
            if en:
                print(f'{sn},{en},ferry')
