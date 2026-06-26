dim = [
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
        12288,
        16384,
        24576,
        32768,
        49152,
]
N=4096
shapes = [
        (i, N, j)
        for i in dim
        for j in dim
        if [i, j] not in [[49152, 32768], [32768, 49152]]
    ]

shapes = sorted(shapes, key=lambda x: x[0] * x[1] * x[2], reverse=True)

print(shapes[0:10])
