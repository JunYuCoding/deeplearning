class HH():
    def __init__(self):
        pass
    def __call__(self, data):
        print("call__",data)
    def forward(self,x):
        return "f"+x

class YY(HH):
    def __init__(self):
        super(YY, self).__init__()
    def forward(self,x):
        return "forward"+x
yy = YY()
yy("哈哈哈")