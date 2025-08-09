# Disclamer!
While yes, you are permitted to use this at your will, I would really appreciate it if you would tag me / credit me.
Thanking you / you lot in advance!!
___
# Overview

1. The `Expression.eval_expr(...)` adds `var: int` which can be set as 1 or 2, indicating the number of variables.
For a single variable experssion, the var is by default set to 1.
If you want two variables, then: `Expression.eval_expr(... , var = 2)`

2. The graphing system now has converged to only one function: `Graphing.plot(...)` for better usage.
Herein, centered, static and speed are set to False, False and 50 respectively.
For two var expressions, z is also required but by default is set to `None`.
___
## Format
> This is for version 1.1 (1_1)

```python
expr = lambda x: (x**2) + 3*x + 5
x, y = Expression.eval_expr(expr) # params for negative and loops
Graphing.plot(x, y) # animated uncentered
Graphing.plot(x, y, static=True) # static uncentered
----//-------(x, b, ..., centered = True) # ... centered
```

> Upcoming v1.2

`v1.2 implements new features like math and physics concepts.`

`Things like projectile path, complex operations and more  will be added.`
- [x] Ideas on what to add
- [x] Implementation and practical math
- [x] Developing / coding
- [ ] Testing, formatting and compatibility checking
- [ ] Documentation (docstrings, repo, etc...)
- [ ] Final thoughts (small tweaks)
- [ ]  Release!!!

___
## Outputs:


With `log10(x) + 2`:
`Warning at x=0: math domain error; setting value=1
[0, 1, 2, 3, ... 99, 100] [1.0, 2.0, 2.3010299956639813, ... 3.9956351945975497, 4.0]
`
Here, the first list is `x` and the next one is `y`.
___
## Purpose

When I was in school, it was hard to visulise graphs for many of the students. And when we did plot them, it was only for 2 or 5 points.
This was made in nostalgia such that one can actually see what is happenning!
