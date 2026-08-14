import re

log = [['Datatype: Simulation'], ['Broadband Filter', 10, 12]]


def applyFilter(f, filter, parameters):
    f.write(parameters[0] + " = preproc.filter_" + filter + '(' + ', '.join(str(e) for e in parameters) + ')')
    print(parameters[0] + " = preproc.filter_" + filter + "(", end='')
    print(*parameters, sep=", ", end='')
    print(')')
    # data = preproc.filter_broadband(data, LowCutOff, HighCutOff, data.get_sampling_rate(), lineNoiseFreq)


def existingImports(content):
    prog = re.compile(r'import .+')
    lines = content.split("\n")
    imports = []
    for line in lines:
        if prog.match(line):
            lib = line[7:].replace(' ', '').split("as")
            if len(lib) == 1:
               lib.append('/')
            imports.append(lib)
    return imports


def manageImports(content, imports, log):
    add = []
    for log_item in log:
        exist = False
        for i in imports:
            if log_item[1] in i:
                exist = True
                break
        if not exist:
            add.append(log_item[1])
    for elt in add:
        content = 'import ' + elt + '\n' + content
    return content


def __getdatatype(string):
    parts = string.split(': ')
    if len(parts) == 2:
        new_string = f'"{parts[0]}": "{parts[1]}"'
        return new_string
    return None

def jsonfilename(filename):
    if not filename.endswith('.json'):
        root = filename.split('.')[0]
        filename = root + '.json'
    return filename



def generateJson(logger, filename='data.json'):
    filename = jsonfilename(filename)
    content = '{\n'
    tabs = 1
    datatype = __getdatatype(logger[0][0])
    if not datatype:
        raise Exception('The logger format is not correct.')
    content += '\t' + __getdatatype(logger[0][0]) + ',\n'
    for i in range(1, len(logger)):
        multiple = False
        op = logger[i]
        content += '\t'
        content +='\"' + op[0] + "\": "
        if len(op) != 2:
            content += '[\n'
            tabs += 1
            multiple = True
        for j in range(1, len(op)):
            if multiple:
                content += '\t' * tabs
            content += "\"" + str(op[j]) + "\""
            if j == len(op) - 1 and multiple:
                tabs -= 1
                content += '\n' + '\t' * tabs + ']'
            elif j != len(op) - 1:
                content += ','
                content += '\n'
        if i != len(logger) - 1:
            content += ','
        content += '\n'

    content += '}'
    with open(filename, "w+") as file:
        file.write(content)
    print("The new json file named "+ filename + " is ready.")
