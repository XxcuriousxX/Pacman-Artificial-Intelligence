"""CSP (Constraint Satisfaction Problems) problems and solvers. (Chapter 6)"""

import itertools
import random
import re
import string
from collections import defaultdict, Counter
from functools import reduce
from operator import eq, neg

from sortedcontainers import SortedSet

import search
from utils import argmin_random_tie, count, first, extend


class CSP(search.Problem):
    """This class describes finite-domain Constraint Satisfaction Problems.
    A CSP is specified by the following inputs:
        variables   A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b

    In the textbook and in most mathematical definitions, the
    constraints are specified as explicit pairs of allowable values,
    but the formulation here is easier to express and more compact for
    most cases (for example, the n-Queens problem can be represented
    in O(n) space using this notation, instead of O(n^4) for the
    explicit representation). In terms of describing the CSP as a
    problem, that's all there is.

    However, the class also supports data structures and methods that help you
    solve CSPs by calling a search function on the CSP. Methods and slots are
    as follows, where the argument 'a' represents an assignment, which is a
    dict of {var:val} entries:
        assign(var, val, a)     Assign a[var] = val; do other bookkeeping
        unassign(var, a)        Do del a[var], plus other bookkeeping
        nconflicts(var, val, a) Return the number of other variables that
                                conflict with var=val
        curr_domains[var]       Slot: remaining consistent values for var
                                Used by constraint propagation routines.
    The following methods are used only by graph_search and tree_search:
        actions(state)          Return a list of actions
        result(state, action)   Return a successor of state
        goal_test(state)        Return true if all constraints satisfied
    The following are just for debugging purposes:
        nassigns                Slot: tracks the number of assignments made
        display(a)              Print a human-readable representation
    """

    def __init__(self, variables, domains, neighbors, constraints, C):
        """Construct a CSP problem. If variables is empty, it becomes domains.keys()."""
        super().__init__(())
        variables = variables or list(domains.keys())
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.curr_domains = None
        # visited nodes
        self.nassigns = 0
        self.conflict_set = {} #dictionary which stores the conflict set of each variable for fc - cbj
        self.prev_conflict_set = [] # we store the conflict set from the variable that causes dead-end
        self.deadend = None  # we save the dead end variable in fc - cbj
        # initializating the conflict set array
        for x in self.variables:
            self.conflict_set[x]=[]
        # --------------------------
        # keep track of total checks for each algo
        self.totchecks=0
        # dict for later use in dom / wdeg heuristic
        # we initializating weights from constraints to 1
        self.weight = {}
        for each in C.keys():
            self.weight[(each[0],each[1])] = 1



    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""
        assignment[var] = val
        self.nassigns += 1

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

    def is_consistent(self, var, val, assignment):    # made for CBJ
        for Xi in self.neighbors[var]:
            if Xi in assignment:
                if not self.constraints(var, val, Xi, assignment[Xi]):
                    # if Xi not in self.curr_conflicts[var]:
                    self.curr_conflicts[var].append(Xi)
                    return False
        return True

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        # Subclasses may implement this more efficiently
        def conflict(var2):
            return var2 in assignment and not self.constraints(var, val, var2, assignment[var2])

        return count(conflict(v) for v in self.neighbors[var])

    def display(self, assignment):
        """Show a human-readable representation of the CSP."""
        # Subclasses can print in a prettier way, or display with a GUI
        print(assignment)

    # These methods are for the tree and graph-search interface:

    def actions(self, state):
        """Return a list of applicable actions: non conflicting
        assignments to an unassigned variable."""
        if len(state) == len(self.variables):
            return []
        else:
            assignment = dict(state)
            var = first([v for v in self.variables if v not in assignment])
            return [(var, val) for val in self.domains[var]
                    if self.nconflicts(var, val, assignment) == 0]

    def result(self, state, action):
        """Perform an action and return the new state."""
        (var, val) = action
        return state + ((var, val),)

    def goal_test(self, state):
        """The goal is to assign all variables, with all constraints satisfied."""
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))

    # These are for constraint propagation

    def support_pruning(self):
        """Make sure we can prune values from domains. (We want to pay
        for this only if we use it.)"""
        if self.curr_domains is None:
            self.curr_domains = {v: list(self.domains[v]) for v in self.variables}

    def suppose(self, var, value):
        """Start accumulating inferences from assuming var=value."""
        self.support_pruning()
        removals = [(var, a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return removals

    def prune(self, var, value, removals):
        """Rule out var=value."""
        self.curr_domains[var].remove(value)
        if removals is not None:
            removals.append((var, value))

    def choices(self, var):
        """Return all values for var that aren't currently ruled out."""
        return (self.curr_domains or self.domains)[var]

    def infer_assignment(self):
        """Return the partial assignment implied by the current inferences."""
        self.support_pruning()
        return {v: self.curr_domains[v][0]
                for v in self.variables if 1 == len(self.curr_domains[v])}

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        for B, b in removals:
            self.curr_domains[B].append(b)

    # This is for min_conflicts search

    def conflicted_vars(self, current):
        """Return a list of variables in current assignment that are in conflict"""
        return [var for var in self.variables
                if self.nconflicts(var, current[var], current) > 0]


# ______________________________________________________________________________
# Constraint Propagation with AC3


def no_arc_heuristic(csp, queue):
    return queue


def dom_j_up(csp, queue):
    return SortedSet(queue, key=lambda t: neg(len(csp.curr_domains[t[1]])))


def AC3(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    """[Figure 6.3]"""

    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        revised, checks = revise(csp, Xi, Xj, removals, checks)
        if revised:
            if not csp.curr_domains[Xi]:
                return False, checks  # CSP is inconsistent
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
    return True, checks  # CSP is satisfiable



def revise(csp, Xi, Xj, removals, checks=0):
    """Return true if we remove a value."""
    revised = False
    for x in csp.curr_domains[Xi][:]:
        conflict = True
        for y in csp.curr_domains[Xj]:
            if csp.constraints(Xi, x, Xj, y):
                conflict = False
            checks += 1
            if not conflict:
                break
        if conflict:
            csp.prune(Xi, x, removals)
            revised = True
    # we check if domains list for our current variable is empty
    # and we increase weight for current variable
    if not csp.curr_domains[Xi]:
        csp.weight[(Xi,Xj)] += 1

    return revised, checks


# Constraint Propagation with AC3b: an improved version
# of AC3 with double-support domain-heuristic

def AC3b(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()

        # Si_p values are all known to be supported by Xj
        # Sj_p values are all known to be supported by Xi
        # Dj - Sj_p = Sj_u values are unknown, as yet, to be supported by Xi
        Si_p, Sj_p, Sj_u, checks = partition(csp, Xi, Xj, checks)
        if not Si_p:
            csp.weight[(Xi,Xj)] += 1
            csp.weight[(Xj,Xi)] += 1
            return False, checks  # CSP is inconsistent
        revised = False
        for x in set(csp.curr_domains[Xi]) - Si_p:
            csp.prune(Xi, x, removals)
            revised = True
        if revised:
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
        if (Xj, Xi) in queue:
            if isinstance(queue, set):
                # or queue -= {(Xj, Xi)} or queue.remove((Xj, Xi))
                queue.difference_update({(Xj, Xi)})
            else:
                queue.difference_update((Xj, Xi))
            # the elements in D_j which are supported by Xi are given by the union of Sj_p with the set of those
            # elements of Sj_u which further processing will show to be supported by some vi_p in Si_p
            for vj_p in Sj_u:
                for vi_p in Si_p:
                    conflict = True
                    if csp.constraints(Xj, vj_p, Xi, vi_p):
                        conflict = False
                        Sj_p.add(vj_p)
                    checks += 1
                    if not conflict:
                        break
            revised = False
            for x in set(csp.curr_domains[Xj]) - Sj_p:
                csp.prune(Xj, x, removals)
                revised = True
            if revised:
                for Xk in csp.neighbors[Xj]:
                    if Xk != Xi:
                        queue.add((Xk, Xj))
    return True, checks  # CSP is satisfiable


def partition(csp, Xi, Xj, checks=0):
    Si_p = set()
    Sj_p = set()
    Sj_u = set(csp.curr_domains[Xj])
    for vi_u in csp.curr_domains[Xi]:
        conflict = True
        # now, in order to establish support for a value vi_u in Di it seems better to try to find a support among
        # the values in Sj_u first, because for each vj_u in Sj_u the check (vi_u, vj_u) is a double-support check
        # and it is just as likely that any vj_u in Sj_u supports vi_u than it is that any vj_p in Sj_p does...
        for vj_u in Sj_u - Sj_p:
            # double-support check
            if csp.constraints(Xi, vi_u, Xj, vj_u):
                conflict = False
                Si_p.add(vi_u)
                Sj_p.add(vj_u)
            checks += 1
            if not conflict:
                break
        # ... and only if no support can be found among the elements in Sj_u, should the elements vj_p in Sj_p be used
        # for single-support checks (vi_u, vj_p)
        if conflict:
            for vj_p in Sj_p:
                # single-support check
                if csp.constraints(Xi, vi_u, Xj, vj_p):
                    conflict = False
                    Si_p.add(vi_u)
                checks += 1
                if not conflict:
                    break
    return Si_p, Sj_p, Sj_u - Sj_p, checks


# Constraint Propagation with AC4

def AC4(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    support_counter = Counter()
    variable_value_pairs_supported = defaultdict(set)
    unsupported_variable_value_pairs = []
    checks = 0
    # construction and initialization of support sets
    while queue:
        (Xi, Xj) = queue.pop()
        revised = False
        for x in csp.curr_domains[Xi][:]:
            for y in csp.curr_domains[Xj]:
                if csp.constraints(Xi, x, Xj, y):
                    support_counter[(Xi, x, Xj)] += 1
                    variable_value_pairs_supported[(Xj, y)].add((Xi, x))
                checks += 1
            if support_counter[(Xi, x, Xj)] == 0:
                csp.prune(Xi, x, removals)
                revised = True
                unsupported_variable_value_pairs.append((Xi, x))
        if revised:
            if not csp.curr_domains[Xi]:
                return False, checks  # CSP is inconsistent
    # propagation of removed values
    while unsupported_variable_value_pairs:
        Xj, y = unsupported_variable_value_pairs.pop()
        for Xi, x in variable_value_pairs_supported[(Xj, y)]:
            revised = False
            if x in csp.curr_domains[Xi][:]:
                support_counter[(Xi, x, Xj)] -= 1
                if support_counter[(Xi, x, Xj)] == 0:
                    csp.prune(Xi, x, removals)
                    revised = True
                    unsupported_variable_value_pairs.append((Xi, x))
            if revised:
                if not csp.curr_domains[Xi]:
                    return False, checks  # CSP is inconsistent
    return True, checks  # CSP is satisfiable


# ______________________________________________________________________________
# CSP Backtracking Search

# Variable ordering


def first_unassigned_variable(assignment, csp):
    """The default variable order."""
    # print(first([var for var in csp.variables if var not in assignment]))

    return first([var for var in csp.variables if var not in assignment])

# implementing heuristic dom/wdeg
def dom_wdeg(assignment, csp):
    # if domains is empty we return the first unassigned variable
    if csp.curr_domains is None:
        return first_unassigned_variable(assignment, csp)

    # setting min variable and min value to infinity
    min_var = float("Inf")
    min_value = float("Inf")
    #sum dictionary to hold the wdeg sum based on paper given.
    sum = {}
    # we iterate through each variable that is not assigned
    # for each of its neighbors who is not assigned as well we
    # increase our sum by the weight of [variable,current neighbor]
    for i in csp.variables:
        if i in assignment:
            continue
        sum[i] = 0
        for j in csp.neighbors[i]:
            if j not in assignment:
                sum[i] += csp.weight[(i, j)]

        # if all neighbors of current variable are assigned we move to the next variable
        if sum[i]==0:
            continue
        # else we update the min_var if we have Found
        # a better ratio dom / wdeg
        l = len(csp.curr_domains[i])
        if  min_value > l /sum[i]:
            min_value = l /sum[i]
            min_var=i

    # we return min var or first_unassigned_variable in case we havent any
    # term in our sum
    if min_var != float("Inf"):
        return min_var
    else:
        return first_unassigned_variable(assignment, csp)




def mrv(assignment, csp):
    """Minimum-remaining-values heuristic."""
    return argmin_random_tie([v for v in csp.variables if v not in assignment],
                             key=lambda var: num_legal_values(csp, var, assignment))


def num_legal_values(csp, var, assignment):
    if csp.curr_domains:
        return len(csp.curr_domains[var])
    else:
        return count(csp.nconflicts(var, val, assignment) == 0 for val in csp.domains[var])


# Value ordering


def unordered_domain_values(var, assignment, csp):
    """The default value order."""
    return csp.choices(var)


def lcv(var, assignment, csp):
    """Least-constraining-values heuristic."""
    return sorted(csp.choices(var), key=lambda val: csp.nconflicts(var, val, assignment))


# Inference


def no_inference(csp, var, value, assignment, removals):
    return True


def forward_checking(csp, var, value, assignment, removals):
    """Prune neighbor values inconsistent with var=value."""
    csp.support_pruning()
    check=0
    for B in csp.neighbors[var]:
        if B not in assignment:
            for b in csp.curr_domains[B][:]:
                check+=1
                if not csp.constraints(var, value, B, b):
                    csp.prune(B, b, removals)
            # we have a failure
            # we check if domains list for variable B is not empty
            # and increase weight of B,var by 1
            if not csp.curr_domains[B]:
                csp.weight[(B,var)] += 1
                return False,check
    return True,check


def mac(csp, var, value, assignment, removals, constraint_propagation=AC3):
    """Maintain arc consistency."""
    return constraint_propagation(csp, {(X, var) for X in csp.neighbors[var]}, removals)


# The search, proper



def backtracking_search(csp, select_unassigned_variable=dom_wdeg,
                        order_domain_values=unordered_domain_values, inference=forward_checking):
    """[Figure 6.5]"""
    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return assignment
        var = select_unassigned_variable(assignment, csp)
        for value in order_domain_values(var, assignment, csp):
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                condition,check=inference(csp, var, value, assignment, removals)
                csp.totchecks+=check
                if condition:
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var, assignment)
        return None

    result = backtrack({})
    assert result is None or csp.goal_test(result)
    return result,csp.totchecks,csp.nassigns
#--------------------------------- Back jumping ----------------------------------
def backjumping_search(csp,select_unassigned_variable = dom_wdeg,order_domain_values = lcv,inference = forward_checking):

    def update_conf(var, csp):
        # we update conflict set for each variable
        # we append the variable in the conflict set if it is not consistent.
        for x in csp.variables:
            if var not in csp.conflict_set[x] and var in csp.neighbors[x]:
                csp.conflict_set[x].append(var)
        return csp.conflict_set

    def merge(var, csp, assigned_var):
        for x in csp.variables:
            tmp = []
            if x == var:
                csp.conflict_set[x] += csp.prev_conflict_set
            for y in assigned_var:
                if y in csp.conflict_set[x]:
                    tmp.append(y)
            csp.conflict_set[x] = tmp
        return csp.conflict_set

    assigned_var = []
    def backjumping(assignment):
        if len(assignment) == len(csp.variables):
            return assignment
        var = select_unassigned_variable(assignment, csp)
        for value in order_domain_values(var, assignment, csp):
            if csp.nconflicts(var, value, assignment) == 0:
                csp.assign(var, value, assignment)
                assigned_var.append(var)
                csp.conflict_set = update_conf(var, csp)
                removals = csp.suppose(var, value)
                condition,check=inference(csp, var, value, assignment, removals)
                csp.totchecks += check
                if condition:
                    result = backjumping(assignment)
                    if result is not None:
                        return result
                csp.restore(removals)

            # we check if we have a dead end with our previous variable
            # and if current variable is in the set of its conflicts.
            # We are trying to find the variable in the conflict set which performs a dead end and it is the deeper in our search tree
            # if we found one variable that satisfy our conditions we merge conflict sets and we turn back at this variable.
            if csp.deadend != None and var in csp.conflict_set[csp.deadend]:
                #we found variable for our conditions so we do the merge between dead end variable set and current variable.
                csp.deadend = None
                csp.conflict_set = merge(var, csp, assigned_var)
            csp.unassign(var, assignment)
            if var in assigned_var:
                del assigned_var[assigned_var.index(var)]
            if csp.deadend != None:
                return None
        # we keep the last variable that cannot test more values as deadend variable
        # and we also keep its conflict set
        csp.deadend = var;
        csp.prev_conflict_set = csp.conflict_set[csp.deadend]
        return None

    result = backjumping({})
    assert result is None or csp.goal_test(result)
    return result,csp.totchecks,csp.nassigns

# ______________________________________________________________________________
# Min-conflicts Hill Climbing search for CSPs


def min_conflicts(csp, max_steps=100000):
    """Solve a CSP by stochastic Hill Climbing on the number of conflicts."""
    # Generate a complete assignment for all variables (probably with conflicts)
    csp.current = current = {}
    for var in csp.variables:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    # Now repeatedly choose a random conflicted variable and change it
    for i in range(max_steps):
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current
        var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    return None,csp.totchecks


def min_conflicts_value(csp, var, current):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))


# ______________________________________________________________________________


def tree_csp_solver(csp):
    """[Figure 6.11]"""
    assignment = {}
    root = csp.variables[0]
    X, parent = topological_sort(csp, root)

    csp.support_pruning()
    for Xj in reversed(X[1:]):
        if not make_arc_consistent(parent[Xj], Xj, csp):
            return None

    assignment[root] = csp.curr_domains[root][0]
    for Xi in X[1:]:
        assignment[Xi] = assign_value(parent[Xi], Xi, csp, assignment)
        if not assignment[Xi]:
            return None
    return assignment


def topological_sort(X, root):
    """Returns the topological sort of X starting from the root.

    Input:
    X is a list with the nodes of the graph
    N is the dictionary with the neighbors of each node
    root denotes the root of the graph.

    Output:
    stack is a list with the nodes topologically sorted
    parents is a dictionary pointing to each node's parent

    Other:
    visited shows the state (visited - not visited) of nodes

    """
    neighbors = X.neighbors

    visited = defaultdict(lambda: False)

    stack = []
    parents = {}

    build_topological(root, None, neighbors, visited, stack, parents)
    return stack, parents


def build_topological(node, parent, neighbors, visited, stack, parents):
    """Build the topological sort and the parents of each node in the graph."""
    visited[node] = True

    for n in neighbors[node]:
        if not visited[n]:
            build_topological(n, node, neighbors, visited, stack, parents)

    parents[node] = parent
    stack.insert(0, node)


def make_arc_consistent(Xj, Xk, csp):
    """Make arc between parent (Xj) and child (Xk) consistent under the csp's constraints,
    by removing the possible values of Xj that cause inconsistencies."""
    # csp.curr_domains[Xj] = []
    for val1 in csp.domains[Xj]:
        keep = False  # Keep or remove val1
        for val2 in csp.domains[Xk]:
            if csp.constraints(Xj, val1, Xk, val2):
                # Found a consistent assignment for val1, keep it
                keep = True
                break

        if not keep:
            # Remove val1
            csp.prune(Xj, val1, None)

    return csp.curr_domains[Xj]


def assign_value(Xj, Xk, csp, assignment):
    """Assign a value to Xk given Xj's (Xk's parent) assignment.
    Return the first value that satisfies the constraints."""
    parent_assignment = assignment[Xj]
    for val in csp.curr_domains[Xk]:
        if csp.constraints(Xj, parent_assignment, Xk, val):
            return val

    # No consistent assignment available
    return None


# ______________________________________________________________________________
# Map Coloring CSP Problems


class UniversalDict:
    """A universal dict maps any key to the same value. We use it here
    as the domains dict for CSPs in which all variables have the same domain.
    >>> d = UniversalDict(42)
    >>> d['life']
    42
    """

    def __init__(self, value): self.value = value

    def __getitem__(self, key): return self.value

    def __repr__(self): return '{{Any: {0!r}}}'.format(self.value)


def different_values_constraint(A, a, B, b):
    """A constraint saying two neighboring variables must differ in value."""
    return a != b


def MapColoringCSP(colors, neighbors):
    """Make a CSP for the problem of coloring a map with different colors
    for any two adjacent regions. Arguments are a list of colors, and a
    dict of {region: [neighbor,...]} entries. This dict may also be
    specified as a string of the form defined by parse_neighbors."""
    if isinstance(neighbors, str):
        neighbors = parse_neighbors(neighbors)
    return CSP(list(neighbors.keys()), UniversalDict(colors), neighbors, different_values_constraint)


def parse_neighbors(neighbors):
    """Convert a string of the form 'X: Y Z; Y: Z' into a dict mapping
    regions to neighbors. The syntax is a region name followed by a ':'
    followed by zero or more region names, followed by ';', repeated for
    each region name. If you say 'X: Y' you don't need 'Y: X'.
    >>> parse_neighbors('X: Y Z; Y: Z') == {'Y': ['X', 'Z'], 'X': ['Y', 'Z'], 'Z': ['X', 'Y']}
    True
    """
    dic = defaultdict(list)
    specs = [spec.split(':') for spec in neighbors.split(';')]
    for (A, Aneighbors) in specs:
        A = A.strip()
        for B in Aneighbors.split():
            dic[A].append(B)
            dic[B].append(A)
    return dic
