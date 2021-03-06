#!/usr/bin/env python
'''
This will generate some sorts of 2 player, one round games.
Then it will try to see how these games compose for parallel
nonlocal players, by trying to come up with an upper bound for
the game's dual

Game is of the form G=(fabxy,pi)
pi dictates the distribution of (a,b) values
the game is won iff fab(a,b) == fxy(x,y) 
'''

import random, itertools, math
from cvxopt import matrix, solvers
from fractions import Fraction
import numpy as np
from z3 import *

def rpartition(n=4):
  #generate a random distribution
  vs = sorted([random.random() for i in range(n-1)])+[1]
  vs2 = [0]*n
  vs2[0] = vs[0]
  for i in range(1,n):
    vs2[i] = vs[i] - vs[i-1]
  return vs2

class Game:
  def __init__(self,fabxy=None,pi=None,G=None,dim=[2,2,2,2]):
    self.dim = dim
    if G is None:
      self.fabxy = fabxy
      if self.fabxy is None:
        self.choice = [random.randint(0,1) for i in xrange(reduce(int.__mul__,self.dim))]
        self.fabxy = lambda a,b,x,y: self.choice[a+self.dim[0]*(b+self.dim[1]*(x+self.dim[2]*y))]

      self.pi = pi
      '''
      if self.pi is None:
        #get a distribution of x,y values
        #we're going to use nice fractions here
        #for now, just a list of choices (these are all the denominators
        # or a 0 for 0 probability)
        pis = [[4,4,4,4],[3,3,3,0],[3,3,0,3],[3,0,3,3],[0,3,3,3],
        [2,2,0,0],[2,0,2,0],[2,0,0,2],[0,0,2,2],[0,2,0,2],[0,2,2,0]]
        self.pi = random.choice(pis)
      '''
      if self.pi is None:
        self.pi = rpartition(reduce(int.__mul__,self.dim[-2:]))

      #our game is 2x2x2x2, entry (a,b,x,y) is P(x,y)*Win(a,b|x,y)
      G = np.zeros(self.dim)
      for (a,b,x,y) in itertools.product(*map(xrange,self.dim)):
        #this is an impossible position
        if self.pi[x*self.dim[2]+y] == 0:
          G[a][b][x][y] = 0
        #this is a winning position
        elif self.fabxy(a,b,x,y):
          G[a][b][x][y] = self.pi[x*self.dim[2]+y]
        else:
          G[a][b][x][y] = 0
      self.G = G
    else:
      self.G = G
    fG = []
    for (a,b,x,y) in itertools.product(*map(xrange,self.dim)):
      fG.append(G[a][b][x][y])
    self.fG =fG

  def __str__(self):
    s = ""
    for a in xrange(self.dim[0]):
      for b in xrange(self.dim[1]):
        for x in xrange(self.dim[2]):
          for y in xrange(self.dim[3]):
            s += "  %3s  "%(str(Fraction(self.G[a][b][x][y]).limit_denominator(20)))
        s += "\n"
    return s

  def __eq__(self,other):
    return self.fG == other.fG

  def __ne__(self,other):
    return self.fG != other.fG

  def __hash__(self):
    s = ""
    for (a,b,x,y) in itertools.product(*map(xrange,self.dim)):
      s += str(self.G[a][b][x][y])
    return hash(s)

  def __mul__(self,other):
    if other.__class__ in [int,float]:
      return self.mulc(other)
    return self.mulg(other)

  def mulc(self,other):
    nG = np.zeros(self.dim)
    for (a,b,x,y) in itertools.product(*map(xrange,self.dim)):
        nG[a][b][x][y] = self.G[a][b][x][y]*other
    return Game(G=nG,dim=self.dim)

  def mulg(self,other):
    nG = np.zeros(map(int.__mul__,self.dim,other.dim))
    for (a,b,x,y) in itertools.product(*map(xrange,self.dim)):
      for (a_,b_,x_,y_) in itertools.product(*map(xrange,other.dim)):
        nG[a_*self.dim[0]+a][b_*self.dim[1]+b][x_*self.dim[2]+x][y_*self.dim[3]+y] = \
          self.G[a][b][x][y]*other.G[a_][b_][x_][y_]
    return Game(G=nG,dim=map(int.__mul__,self.dim,other.dim))

  def __add__(self,other):
    nG = np.zeros(self.dim)
    for (a,b,x,y) in itertools.product(*map(xrange,self.dim)):
        nG[a][b][x][y] = self.G[a][b][x][y]+other.G[a][b][x][y]
    return Game(G=nG,dim=self.dim)

  def __sub__(self,other):
    nG = np.zeros(self.dim)
    for (a,b,x,y) in itertools.product(*map(xrange,self.dim)):
        nG[a][b][x][y] = self.G[a][b][x][y]-other.G[a][b][x][y]
    return Game(G=nG,dim=self.dim)

  #approximates this game using a convex combination of games from
  #using, such that the coefficients of the games in using are minimized
  def approx(self,using):
#    if self.choice == [0]*len(self.choice):
#      return [0]*len(using)
    b = matrix(map(lambda x:-x,self.fG) + [0]*len(using))
    c = matrix( [1.0]*len(using) )
    tA = []
    i = 0
    for u in using:
      me = [0]*len(using)
      me[i] = 1
      i += 1
      tA.append(map(lambda x:-x, u.fG + me))
    solvers.options['LPX_K_MSGLEV'] = 0
    solvers.options['show_progress'] = False
    s = solvers.lp(c,matrix(tA),b,solver='glpk')
    return [v for v in s['x']]
    return [Fraction(v).limit_denominator(20).numerator*1.0/
      Fraction(v).limit_denominator(20).denominator for v in s['x']]

#find a classical strategy (ie no communication from A to B) for g
#this is not going to be fast...
def cstrat(g):
  asts = itertools.product(xrange(g.dim[0]),repeat=g.dim[2])
  best = (0,None)
  for ast in asts:
    bsts = itertools.product(xrange(g.dim[1]),repeat=g.dim[3])
    for bst in bsts:
      v = 0
      for (x,y) in itertools.product(*map(xrange,g.dim[-2:])):
        v += g.G[ast[x]][bst[y]][x][y]
      if v > best[0]:
        best = (v,(ast,bst))      
  return best

def csstrat(g):
  asts = itertools.product(range(g.dim[0])[::-1],repeat=g.dim[2])
  best = (0,None)
  for ast in asts:
    v = 0
    for (x,y) in itertools.product(*map(xrange,g.dim[-2:])):
      v += g.G[ast[x]][ast[y]][x][y]
    if v > best[0]:
      print v,ast
      best = (v,(ast,ast))      
  return best

def z3strat(g):
  ast = [Bools(' '.join(["ast%d_%d"%(x,d) for d in xrange(g.dim[0])])) for x in xrange(g.dim[0])]
  s = Solver()
  v = 0
  for xast in ast:
    for (a,b) in itertools.combinations(xast,2):
      s.add(Not(And(a,b)))
    s.add(Or(xast))
  for (a,b,x,y) in itertools.product(*map(xrange,g.dim)):
    v += If(ast[x][a], If(ast[y][b], g.G[a][b][x][y] ,0) ,0)
  return s,ast,v

def tstrat(s,g):
  (ast,bst) = s
  v = 0
  for (x,y) in itertools.product(*map(xrange,g.dim[-2:])):
    v += g.G[ast[x]][bst[y]][x][y]
  return v

def getleft(dim=[2,2,2,2]):
  left = []
  #[[0]*dim**2 for i in dim]
  for p in xrange(dim[2]):
    pi = [0 if ((x/dim[3]) != p) else 1 for x in xrange(dim[2]*dim[3])]
    for aa in itertools.product(range(dim[3]),repeat=dim[0]):
      left.append(Game(dim=dim,fabxy = lambda a,b,x,y: 1 if y == aa[a] else 0,pi=pi))
  return left

def getright(dim=[2,2,2,2]):
  right= []
  #[[0]*dim**2 for i in dim]
  for p in xrange(dim[3]):
    pi = [0 if ((x%dim[2]) != p) else 1 for x in xrange(dim[2]*dim[3])]
    for bb in itertools.product(range(dim[2]),repeat=dim[1]):
      right.append(Game(dim=dim,fabxy = lambda a,b,x,y: 1 if x == bb[b] else 0,pi=pi))
  return right

left = getleft()

right = getright()

using = left+right

def separate(used,results,side1,side2):
  assert(used == side1 + side2)
  return (reduce(lambda x,y:x+y,map(lambda x,y:x*y,side1,results[0:len(side1)])),
    reduce(lambda x,y:x+y,map(lambda x,y:x*y,side2,results[len(side1):])))

#helper functions for now...
imp = lambda a,b: True if (not(a) or b) else False
ff = lambda x: True if x else False
new = lambda a,b,x,y: (imp(y!=0,a==y) and imp(x!=0,b==x)) and imp((y == 0) and (x == 0),(ff(a) != ff(b))) and not (ff(x) and ff(y))
