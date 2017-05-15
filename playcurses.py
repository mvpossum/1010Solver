import curses
stdscr = curses.initscr()

begin_x = 0; begin_y = 0
height = 25; width = 60
win = curses.newwin(height, width, begin_y, begin_x)

#  These loops fill the pad with letters; this is
# explained in the next section
#~ for y in range(0, height):
    #~ for x in range(0, width):
        #~ try:
            #~ win.addch(y,x, ord('a') + (x*x+y*y) % 26)
        #~ except curses.error:
            #~ pass
        
stdscr.refresh() 
   
curses.mousemask(curses.ALL_MOUSE_EVENTS)#curses.BUTTON1_PRESSED)#curses.ALL_MOUSE_EVENTS)
while 1:
    win.keypad(1)
    curses.halfdelay(1)
    ch = win.getch()
    win.addstr(0, 0, str(ch))
    if ch == curses.KEY_MOUSE:
        _, x, y, z, bstate =  curses.getmouse()
        win.addstr(4,0, "%s" % ' '.join(str((x, y))))
    #return ch
    win.refresh()

curses.endwin()
