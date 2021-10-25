from py2puml.py2puml import py2puml

with open('py2puml/py2puml.domain.puml', 'w') as puml_file:
    ss = [s.replace("src.", "") for s in py2puml('src', 'src')]
    for i in range(1, len(ss)-1):
        s = ss[i]
        if s.startswith(" "):
            ss[i] = "+" + s
    puml_file.writelines(ss)
