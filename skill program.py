class Person:
    def __init__(self, n, o):
        self.name = n
        self.occupation = o

    def wokr(self):
        if(self.occupation) == "professor":
            print(self.name ,"give lecture")
        else:
            print(self.name,"don't give the lecture")

    def speaks(self):
        print(self.name,"says hi")

p=Person("pradeep","professor")
p.wokr()
p.speaks()
