from z3 import *
r1, r2, r3 = Bools('r1 r2 r3') #true is lady, false is tiger
s1, s2, s3 = Bools('s1 s2 s3') #whether the sign is true
s = Solver()
s.add(Or(And(r1, Not(r2), Not(r3)), And(Not(r1), r2, Not(r3)), And(Not(r1), Not(r2), r3)))
# lady is in one of the room
s.add(s1==Not(r1), s2==r2, s3==Not(r2))
s.add(Or(And(s1, Not(s2), Not(s3)), And(Not(s1), s2, Not(s3)), And(Not(s1), Not(s2), s3), And(Not(s1), Not(s2), Not(s3))))
# at most one sign is true

s.push()
s.add(Not(r1))
print(s.check())# lady is in room1
s.pop()
print(s.check())
print(s.model())