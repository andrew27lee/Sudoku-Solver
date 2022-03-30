class MyHoughPLines:
    def __init__(self, line_raw, ratio=1):
        line_raw = line_raw[0]
        self.x1 = int(line_raw[0]*ratio)
        self.y1 = int(line_raw[1]*ratio)
        self.x2 = int(line_raw[2]*ratio)
        self.y2 = int(line_raw[3]*ratio)
        self.isMerged = False
        self.number_of_merged = 1
    
    def get_limits(self):
        return self.x1, self.y1, self.x2, self.y2
