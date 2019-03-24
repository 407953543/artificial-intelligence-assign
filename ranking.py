from z3 import *
L, B, J, M = Ints('L B J M')
LB, JB, BB, MB = Bools('LB BB JB MB') #whether biology major
s = Solver()
s.add(L>0, L<5, B>0, B<5, J>0, J<5, M>0, M<5)
s.add(L!=B, L!=J, L!=M, B!=J, B!=M, J!=M)
s.add(L!=B+1, B!=L+1) #lisa is not next to bob
s.add(Or(And(L-J==1, LB), And(B-J==1, BB), And(M-J==1, MB))) #jim is ahead of a biology major
s.add(J==B+1) #bob is adead of jim
s.add(Or(LB, MB)) #one of women is biology major
s.add(Or(L==1, M==1)) #one of women ranked first
print(s.check())
print(s.model())