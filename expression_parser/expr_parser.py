import numpy as np

class ExprChr():
    
    def __init__(self, char: str, x: int, y: int, w: int, h: int):
        self.char = char
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        
    def center(self) -> tuple[int, int]:
        return ((self.x+self.w)//2, (self.y+self.h)//2)

    def avgRadius(self) -> int:
        return (self.w>>2)+(self.h>>2)
    
    def __eq__(self, value):
        return self.char == value.char and self.x == value.x and self.y == value.y and self.w == value.w and self.h == value.h

    def __ne__(self, value):
        return not self.__eq__(value)

def parse(chars: list[ExprChr], dst_fact:float = 0.5) -> list[ExprChr]:
    closest: dict[ExprChr, list[ExprChr]] = {}
    mean_avg_bb_size = np.mean(list(map(lambda x: x.avgRadius(), chars)))
    for char1 in chars:
        closest[char1] = sorted([c for c in chars if c != char1], key=lambda x: squardLen(char1.center()-x.center())<mean_avg_bb_size)
        print(char1+': '+closest[char1])
    
    # fraction checking
    for c, cs in closest.items():
        if c.char == '/' and len(cs) >= 2:
            # check y dir
            # check above, below
            # save state
            # check multi symbol
            # construct the two expression
            # replace the original with the final one
            for cchr in cs:
                diff = cchr.y-c.y
                if abs(diff) < dst_fact*mean_avg_bb_size:
                    pass

    return None

def centerChr(exprChr: tuple[str, int, int, int, int]) -> tuple[int, int]:
    _, x, y, w, h = exprChr
    return center(x, y, w, h)         

def center(x:int, y:int, w:int, h:int) -> tuple[int, int]:
    return ((x+w)/2, (y+h)/2)

def squardLen(x: int, y: int) -> int:
    return x**2+y**2