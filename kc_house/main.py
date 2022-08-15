
class Greeting:
    def __init__(self, name):
        self.name = name

    def say_hello(self):
        print(f'hello dai ca {self.name}')


class Tmp:
    def __init__(self, name):
        self.name = name

    def show_name(self):
        print(f'hello dai ca {self.name}')


if __name__ == '__main__':
    tmp = Greeting("Linh Nguyen")
    tmp.say_hello()

    t = Tmp("Linh Nguyen")
    t.show_name()
