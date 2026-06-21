def ptx_simulate(x0, x1, x2, x3):
    r0 = list(x0)
    r1 = list(x1)
    r2 = list(x2)
    r3 = list(x3)
    
    # Step 1
    e1 = [0]*4
    e3 = [0]*4
    for t in range(4):
        p1 = (t & 1) != 0
        s1 = r0[t] if p1 else r1[t]
        s3 = r2[t] if p1 else r3[t]
        
        neighbor = t ^ 1
        p1_n = (neighbor & 1) != 0
        s1_n = r0[neighbor] if p1_n else r1[neighbor]
        s3_n = r2[neighbor] if p1_n else r3[neighbor]
        
        e1[t] = s1_n
        e3[t] = s3_n

    for t in range(4):
        p1 = (t & 1) != 0
        if p1:
            r0[t] = e1[t]
            r2[t] = e3[t]
        else:
            r1[t] = e1[t]
            r3[t] = e3[t]
            
    # Step 2
    e2 = [0]*4
    e3_new = [0]*4
    for t in range(4):
        p2 = (t & 2) != 0
        s2 = r0[t] if p2 else r2[t]
        s3_v = r1[t] if p2 else r3[t]
        
        neighbor = t ^ 2
        p2_n = (neighbor & 2) != 0
        
        s2_n = r0[neighbor] if p2_n else r2[neighbor]
        s3_n_v = r1[neighbor] if p2_n else r3[neighbor]
        
        e2[t] = s2_n
        e3_new[t] = s3_n_v

    for t in range(4):
        p2 = (t & 2) != 0
        if p2:
            r0[t] = e2[t]
            r1[t] = e3_new[t]
        else:
            r2[t] = e2[t]
            r3[t] = e3_new[t]
            
    return r0, r1, r2, r3

x0 = ["T0:r0", "T1:r0", "T2:r0", "T3:r0"]
x1 = ["T0:r1", "T1:r1", "T2:r1", "T3:r1"]
x2 = ["T0:r2", "T1:r2", "T2:r2", "T3:r2"]
x3 = ["T0:r3", "T1:r3", "T2:r3", "T3:r3"]

y0, y1, y2, y3 = ptx_simulate(x0, x1, x2, x3)

for t in range(4):
    print(f"Thread {t}: y0={y0[t]}, y1={y1[t]}, y2={y2[t]}, y3={y3[t]}")
