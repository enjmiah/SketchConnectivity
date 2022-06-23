# CmakeLists.txt in Test

INCLUDE_DIRECTORIES(${Cornucopia_SOURCE_DIR}/Cornucopia)
INCLUDE_DIRECTORIES(${Cornucopia_SOURCE_DIR}/DemoUI)

FILE(GLOB CornucopiaTest_CPP "*.cpp")
FILE(GLOB CornucopiaTest_H "*.h")

LIST(APPEND CornucopiaTest_Sources ${CornucopiaTest_CPP} ${CornucopiaTest_H})

ADD_EXECUTABLE(CornucopiaTest ${CornucopiaTest_Sources})

TARGET_LINK_LIBRARIES(CornucopiaTest Cornucopia ${QT_LIBRARIES})
