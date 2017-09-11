def print_table(tags, values, header=False, width=42):
    ''' Print a pretty table =) '''

    tags_maxlen = max(len(tag) for tag in tags)
    values_maxlen = max(len(value) for value in values)

    max_width = max(width, tags_maxlen + values_maxlen)

    print()
    if header:
        print(header)
    print((2 + max_width) * '-')
    for tag, value in zip(tags, values):
        num_spaces = 2 + values_maxlen - len(value)
        string_right = '{:{n}}{}'.format('|', value, n=num_spaces)
        num_spaces = 2 + max_width - len(tag) - len(string_right)
        print(''.join((tag, ' ' * num_spaces, string_right)))
    print((2 + max_width) * '-')
