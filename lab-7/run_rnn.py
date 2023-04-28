import dearpygui.dearpygui as dpg
from rnn import SAVED_IMAGES, RNN


def main():
    net = RNN(SAVED_IMAGES)
    check_image = [0] * 15

    dpg.create_context()

    with dpg.theme() as white_cell:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, [255, 255, 255])

    with dpg.theme() as black_cell:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, [0, 0, 0])

    with dpg.theme() as blue_window:
        with dpg.theme_component(dpg.mvWindowAppItem):
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, [0, 51, 153, 255])
            dpg.add_theme_color(dpg.mvThemeCol_Border, [49, 109, 228, 255])
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, [49, 109, 228, 255])

    for i in range(len(SAVED_IMAGES)):
        with dpg.window(
                label=F'IMAGE {i + 1}',
                pos=(100 * i, 0),
                no_resize=True,
                no_move=True,
                no_close=True,
                no_collapse=True,
        ) as w:
            dpg.bind_item_theme(w, blue_window)
            with dpg.table(header_row=False):
                for j in range(3):
                    dpg.add_table_column()

                for row in range(5):
                    with dpg.table_row():
                        for col in range(3):
                            cell = dpg.add_text('o')
                            if SAVED_IMAGES[i][col * 5 + row] > 0:
                                dpg.bind_item_theme(cell, white_cell)
                                continue
                            dpg.bind_item_theme(cell, black_cell)

    def check_table_callback(sender, app_data, user_data) -> None:
        check_image[user_data] = 1 if app_data else -1

    def check_result(sendes, app_data, user_data) -> None:
        res = net.recover_image(check_image, 'SYNC' if user_data else 'ASYNC')
        dpg.configure_item('RESULT', default_value=res)

    with dpg.theme() as check_window:
        with dpg.theme_component(dpg.mvWindowAppItem):
            dpg.add_theme_color(dpg.mvThemeCol_Text, [255, 255, 255])
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, [49, 109, 228, 255])

    with dpg.theme() as check_cell:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, [255, 255, 255])

    with dpg.theme() as check_buttons:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Button, [49, 109, 228, 255])

    with dpg.window(
            label='IMAGE CHECK',
            pos=(50, 150),
            no_resize=True,
            no_move=True,
            no_close=True,
            no_collapse=True,
            width=200,
    ) as check_w:
        dpg.bind_item_theme(check_w, check_window)

        with dpg.table(header_row=False, width=100):
            for j in range(3):
                dpg.add_table_column()

            for row in range(5):
                with dpg.table_row():
                    for col in range(3):
                        cell = dpg.add_selectable(label='o',
                                                  callback=check_table_callback,
                                                  user_data=col * 5 + row)
                        dpg.bind_item_theme(cell, check_cell)

        with dpg.table(header_row=False, width=190):
            for j in range(2):
                dpg.add_table_column()

            with dpg.table_row():
                b1 = dpg.add_button(label='SYNC', width=80, user_data=True, callback=check_result)
                b2 = dpg.add_button(label='ASYNC', width=80, user_data=False, callback=check_result)

                dpg.bind_item_theme(b1, check_buttons)
                dpg.bind_item_theme(b2, check_buttons)

        dpg.add_input_text(hint='RESULT', enabled=False, width=115, tag='RESULT')

    dpg.create_viewport(
        title='HOPFIELD RNN',
        width=300,
        height=315,
        resizable=False,
    )

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == '__main__':
    main()
