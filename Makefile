CC= g++
RM=rm -f
LINKER_FLAGS=-lpthread
COMPILER_FLAGS=-Ieigen-3.3.7/ -O3
OBJ_NAME=main

SRCS= $(wildcard ./*.cpp)
OBJS=$(subst .cpp,.o,$(SRCS))

run: main
	rm -rf *.o

$(OBJ_NAME):$(OBJS)
	$(CC) $(COMPILER_FLAGS) -o $(OBJ_NAME) $(OBJS) $(LINKER_FLAGS)

%.o: %.cpp
	$(CC) $(COMPILER_FLAGS) -c $<

.depend: $(SRCS)
	$(RM) ./.depend
	$(CC) $(COMPILER_FLAGS) -MM $$^>>./.depend

clean:
	$(RM) *.o $(OBJ_NAME)
