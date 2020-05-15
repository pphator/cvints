class CvintsException(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            result_msg = 'Cvints Error, {0}'.format(self.message)
        else:
            result_msg = 'Cvints Error has been raised'
        return result_msg
