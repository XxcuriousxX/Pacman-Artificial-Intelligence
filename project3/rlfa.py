import sys
import csp



def get_variables(var_filename):
	# # we read var file and return a list with variables

    f=open("../rlfap/"+var_filename, "r")
    text=f.readlines()
    var_domain={}
    variables=[]
    for line in text[1:]:
        str=format(line.strip("\n"))
        x,y=str.split(" ")
        variables.append(int(x))
        var_domain[int(x)]=int(y)

    f.close()

    return variables,var_domain

def get_doms(filename,variables,var_domain):
	# we return a dictionary with var id as the key and
	# a list with all the possible values the variable in key can take as item

    f=open("../rlfap/"+ filename, "r")
    text=f.readlines()
    domains={}
    total={} # dictionary with domain's ID as key and as item a list which contains the domain's values
    for line in text[1:]:
        line=line.strip("\n")
        str=line.split(" ")
        str=[int(i) for i in str]
        for id in str[:1]:
            total[id]=str[2:]

    for var in variables:
        domains[var]=total[var_domain[var]]


    f.close()
    return domains



def ctrfile(filename):
	f=open("../rlfap/"+ filename, "r")
	# we return 2 dicts one for our constraints and one for the neighbors
	# constraints dict has a tuple with vars as key and a tuple with operator and k-value as item
	# neighbors dict has a var id as key and a list with all neighbors of the key var as item
	read=f.readlines()
	constraints={}
	neighbors={}
	for line in read[1:]:
		str=format(line.strip(" "))

		x,y,op,k=str.split(" ")

		x=int(x)
		y=int(y)
		k=int(k)

		if x in neighbors:
			neighbors[x].append(y)
		else:
			neighbors[x] = []
			neighbors[x].append(y)

		if y in neighbors:
			neighbors[y].append(x)
		else:
			neighbors[y] = []
			neighbors[y].append(x)

		constraints[(x,y)]=(k,op)
		constraints[(y,x)]=(k,op)


	f.close()
	return constraints,neighbors



def check_constraints(A,a,B,b):
    # checking if the constraints between 2 variables (A and B) for the corresponding values (a,b) apply
    if (A,B) in constraints:
        k = constraints[(A,B)][0]
        if constraints[(A,B)][1] == '=':
            return (abs(a-b)==k)
        else:
            return (abs(a-b)>k)
    elif (B,A) in constraints:
        k = constraints[(B,A)][0]
        if constraints[(B,A)][1] == '=':
            return (abs(a-b)==k)
        else:
            return (abs(a-b)>k)


if (len(sys.argv) == 1):
    print("")
    print("-Please give <problem-txt> and <algo>")
    print("-Check readme for further explanation!")
    print("ex.  ------>  rlfa.py 6-w2 FC  <---------")
    quit()

instance = sys.argv[1]
variables_file = "var" + instance + ".txt"
domains_file = "dom" + instance + ".txt"
constraints_file = "ctr" + instance + ".txt"

variables,var_to_domain = get_variables(variables_file)
domains = get_doms(domains_file,variables,var_to_domain)
constraints,neighbors = ctrfile(constraints_file)

rlfa_obj = csp.CSP(variables,domains,neighbors,check_constraints,constraints)

if sys.argv[2] == "FC":
	print("FC algorithm - Dom/wdeg heuristic")
	result = csp.backtracking_search(rlfa_obj,select_unassigned_variable=csp.dom_wdeg,order_domain_values=csp.lcv,inference=csp.forward_checking)
	print("Solution:")
	print(result[0])
	print("Number of Checks: %d" % result[1])
	print("Number of visited nodes: %d" % result[2])
if sys.argv[2] == "CBJ":
	print("FC-CBJ algorithm - Dom/wdeg heuristic")
	result = csp.backjumping_search(rlfa_obj,select_unassigned_variable=csp.dom_wdeg,order_domain_values=csp.lcv,inference=csp.forward_checking)
	print("Solution:")
	print(result[0])
	print("Number of Checks: %d" % result[1])
	print("Number of visited nodes: %d" % result[2])
if sys.argv[2] == "MAC":
	print("MAC Algorithm - Dom/wdeg heuristic")
	result = csp.backtracking_search(rlfa_obj, select_unassigned_variable=csp.dom_wdeg,order_domain_values=csp.lcv,inference=csp.mac)
	print("Solution:")
	print(result[0])
	print("Number of Checks: %d" % result[1])
	print("Number of visited nodes: %d" % result[2])
if sys.argv[2] == "MIN":
	print("Min conflicts")
	result = csp.min_conflicts(rlfa_obj)
	print("Solution:")
	print(result[0])
	print("Number of visited nodes: %d" % rlfa_obj.nassigns)
