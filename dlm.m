%状態空間モデルを定義
classdef dlm 
    methods (Static)
        function y = h(x)
            y = x.*x + x.*x.*x;
        end

        function x_ = f(x)
            x_ = x;
        end
    end
end