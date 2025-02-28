def get_task(name, args):
    if name == 'game24':
        from tot.tasks.game24 import Game24Task
        return Game24Task(args)
    elif name == 'text':
        from tot.tasks.text import TextTask
        return TextTask(args)
    elif name == 'crosswords':
        from tot.tasks.crosswords import MiniCrosswordsTask
        return MiniCrosswordsTask(args)
    else:
        raise NotImplementedError