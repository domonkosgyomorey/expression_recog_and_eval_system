def get_center(symbol):
    x, y, w, h = symbol[1:]
    return (x + w/2, y + h/2)

def get_bounds(symbol):
    x, y, w, h = symbol[1:]
    return (x, x + w, y, y + h)

def overlaps_horizontally(symbol1, symbol2, tolerance_factor=0.3):
    x1, x1_end, _, _ = get_bounds(symbol1)
    x2, x2_end, _, _ = get_bounds(symbol2)
    
    # Átfedés toleranciájának számítása a szimbólumok szélességéhez képest
    tolerance = min(symbol1[3], symbol2[3]) * tolerance_factor
    
    return (x1 <= x2_end + tolerance and x2 <= x1_end + tolerance)

def is_fraction_bar(symbol, symbols):
    if symbol[0] != '-':
        return False
    
    bar_x, bar_y, bar_w, bar_h = symbol[1:]
    bar_center_y = bar_y + bar_h/2
    
    # Keressünk szimbólumokat a vonal felett és alatt
    above_symbols = []
    below_symbols = []
    
    for s in symbols:
        if s == symbol:
            continue
            
        s_center = get_center(s)
        if overlaps_horizontally(symbol, s):
            if s_center[1] < bar_y:
                above_symbols.append(s)
            elif s_center[1] > bar_y + bar_h:
                below_symbols.append(s)
    
    return len(above_symbols) > 0 and len(below_symbols) > 0

def get_fraction_parts(fraction_bar, symbols):
    bar_x, bar_y, bar_w, bar_h = fraction_bar[1:]
    
    numerator = []
    denominator = []
    
    for symbol in symbols:
        if symbol == fraction_bar:
            continue
            
        if overlaps_horizontally(fraction_bar, symbol):
            symbol_center = get_center(symbol)
            if symbol_center[1] < bar_y:
                numerator.append(symbol)
            elif symbol_center[1] > bar_y + bar_h:
                denominator.append(symbol)
    
    # Rendezzük a szimbólumokat balról jobbra
    numerator.sort(key=lambda s: get_center(s)[0])
    denominator.sort(key=lambda s: get_center(s)[0])
    
    return numerator, denominator

def parse_expression(symbols):
    # Rendezzük a szimbólumokat x koordináta szerint
    sorted_symbols = sorted(symbols, key=lambda s: s[1])  # x koordináta szerint rendezés
    expression = []
    processed = set()
    
    for i, symbol in enumerate(sorted_symbols):
        if i in processed:
            continue
        
        # Törtvonal feldolgozása
        if symbol[0] == '-' and is_fraction_bar(symbol, sorted_symbols):
            numerator, denominator = get_fraction_parts(symbol, sorted_symbols)
            
            if numerator and denominator:
                num_expr = ''.join(str(s[0]) for s in numerator)
                den_expr = ''.join(str(s[0]) for s in denominator)
                expression.append(f"{num_expr}/{den_expr}")
                
                # Jelöljük a tört minden részét feldolgozottként
                processed.add(i)
                for s in numerator + denominator:
                    processed.add(sorted_symbols.index(s))
        
        # Egyéb szimbólumok feldolgozása
        elif i not in processed:
            expression.append(str(symbol[0]))
            processed.add(i)
    
    return ' '.join(expression)